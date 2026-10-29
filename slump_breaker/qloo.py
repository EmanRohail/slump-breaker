"""The one place the app talks to Qloo.

Everything else calls QlooClient.resolve() and QlooClient.book_recs(), so when
the real API key arrives only this file needs checking.

Verified against the real hackathon API (Oct 2026 probe): GET /search returns
{results: [{name, entity_id, types, tags, popularity, properties}]} and
GET /v2/insights (filter.type=urn:entity:book, signal.interests.entities=<id>)
returns {results: {entities: [...]}} where each book has properties.page_count,
properties.image.url, query.affinity and a "disambiguation" of "<year>, <author>".
Insights books carry no tags by default. Search results are NOT ordered by what
a reader means ("Hades" returns a metal band first), so resolve() picks by exact
name, an optional type hint such as "Hades (game)", and a type priority.

With no API key the client runs in mock mode using fixtures from mock_data.py.
"""
from __future__ import annotations

import html
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Callable, Optional

import requests

from . import mock_data
from .models import Candidate, Entity

CACHE_TTL_SECONDS = 6 * 60 * 60
SEARCH_TAKE = 10
# Box sets, colouring books, guides and other editions that are a poor way back into reading.
_OMNIBUS = re.compile(
    r"(box(ed)?\s*set|omnibus|collection|complete\s+(series|works)|#\s*\d+\s*[-–]\s*\d+"
    r"|(books?|vol(ume)?s?\.?)\s*\d+\s*[-–]\s*\d+|colou?ring|coloring|activity\s+book|sticker"
    r"|study\s+guide|summary\s+of|workbook|journal|notebook|calendar|cliffs?\s*notes|sparknotes|official\s+(guide|companion)|the\s+art\s+of)",
    re.I,
)

# What a reader's hint ("Hades (game)") means in Qloo's entity types.
_HINTS = {
    "game": "videogame", "video game": "videogame", "videogame": "videogame",
    "show": "tv_show", "tv": "tv_show", "tv show": "tv_show", "series": "tv_show", "anime": "tv_show",
    "movie": "movie", "film": "movie",
    "band": "artist", "artist": "artist", "singer": "artist", "musician": "artist", "music": "artist",
    "album": "album", "book": "book", "podcast": "podcast", "brand": "brand", "person": "person",
    "kdrama": "tv_show", "k-drama": "tv_show", "k drama": "tv_show", "cdrama": "tv_show", "c-drama": "tv_show",
    "c drama": "tv_show", "jdrama": "tv_show", "j-drama": "tv_show", "drama": "tv_show", "novel": "book",
    "genre": "genre", "trope": "trope",
}
# "Goblin kdrama" / "Death Note anime": a trailing word that says what kind of title it is. Only words that are
# never part of a real title are used ("Squid Game" and "The Truman Show" must stay whole).
_TRAIL_RE = re.compile(r"^(.+?)[\s-]+(k[\s-]?drama|c[\s-]?drama|j[\s-]?drama|anime|tv series|tv show|video ?game)$", re.I)
_TRAIL_KIND = {"kdrama": "tv_show", "cdrama": "tv_show", "jdrama": "tv_show", "anime": "tv_show",
               "tvseries": "tv_show", "tvshow": "tv_show", "videogame": "videogame"}
# When a name matches several kinds of thing and no hint is given.
_TYPE_PRIORITY = ["tv_show", "movie", "videogame", "artist", "book", "podcast", "brand", "person", "album"]
_HINT_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")

# Short forms people type, expanded token by token ("gta-5" -> "grand theft auto v").
_ALIASES = {
    "gta": "grand theft auto", "cod": "call of duty", "lotr": "the lord of the rings",
    "hp": "harry potter", "got": "game of thrones", "mcu": "marvel cinematic universe",
    "botw": "breath of the wild", "totk": "tears of the kingdom", "rdr": "red dead redemption",
    "rdr2": "red dead redemption 2", "csgo": "counter strike", "lol": "league of legends",
    "wow": "world of warcraft", "ac": "assassins creed", "tlou": "the last of us",
    "bts": "bts", "got7": "got7", "st": "stranger things", "otp": "otp",
}
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}
MIN_SIMILARITY = 0.5
# Among equally close titles, one this far below the most popular is a lesser namesake and loses even if its type
# comes first: "Circe" the artist (0.66) vs Miller's novel (0.9998), "Dune" the 2000 show (0.92) vs the film (0.9996).
# Famous titles sit close together near 1, so the type priority still decides "Hades" (artist 0.96, game 0.94).
POPULARITY_GAP = 0.05
# Qloo names series books "Twisted Love (Twisted, #1)" or "Dune (Dune, #1)"; readers type "Twisted Love".
_SERIES_TAG = re.compile(r"\s*\([^()]*#\s*\d[^()]*\)\s*$")
# Qloo refuses these as a taste signal for book recommendations (seen live: an album and a place gave HTTP 400),
# and they are almost never what a reader means. They are only used if nothing else matches.
_UNUSABLE_TYPES = {"album", "place", "destination", "locality", "country", "city"}
# Words that name a kind of thing, not a title.
_GENERIC = {"kdrama", "k drama", "cdrama", "c drama", "jdrama", "drama", "dramas", "anime", "movie", "movies", "film",
            "films", "series", "show", "shows", "tv", "game", "games", "music", "song", "songs", "book", "books",
            "novel", "novels", "japan", "korea", "china", "india", "pakistan", "romance", "fantasy", "comedy", "horror",
            # topics, hobbies and sports people type instead of a title (they match unrelated places or albums)
            "thriller", "mystery", "murder", "crime", "action", "adventure", "scifi", "sci fi", "science fiction",
            "historical", "history", "period drama", "webtoon", "webtoons", "manga", "manhwa", "donghua",
            "cricket", "football", "soccer", "volleyball", "basketball", "tennis", "badminton", "hockey", "baseball",
            "swimming", "running", "gym", "yoga", "baking", "cooking", "dessert", "desserts", "food", "travel",
            "gaming", "reading", "painting", "drawing", "dancing", "singing", "fashion", "makeup", "photography"}


# Popular shows people call by a short or alternate name; Qloo lists them under the full title.
# Used as an extra search spelling for shows ("Goblin" would otherwise match the anime "Goblin Slayer").
_SHOW_ALIASES = {
    "goblin": ["Guardian: The Lonely and Great God", "Goblin: The Lonely and Great God"],
    "dots": ["Descendants of the Sun"],
    "cloy": ["Crash Landing on You"],
}


def known_alias(name: str, hint: Optional[str]) -> list[str]:
    """Full titles for a short show name (only when the reader means a show or gave no type)."""
    return _SHOW_ALIASES.get(norm_title(name), []) if hint in (None, "tv_show") else []


# A genre word on its own is not something Qloo can use, but readers type them all the time ("kdrama").
# We swap each for a few very well-known titles of that kind, so the taste graph still has something real to work with.
GENRE_ANCHORS = {
    "kdrama": ["Crash Landing on You (show)", "Squid Game (show)"],
    "k drama": ["Crash Landing on You (show)", "Squid Game (show)"],
    "korean drama": ["Crash Landing on You (show)", "Squid Game (show)"],
    "cdrama": ["The Untamed (show)", "Nirvana in Fire (show)"],
    "c drama": ["The Untamed (show)", "Nirvana in Fire (show)"],
    "chinese drama": ["The Untamed (show)", "Nirvana in Fire (show)"],
    "historical cdrama": ["Nirvana in Fire (show)", "The Story of Yanxi Palace (show)"],
    "historical drama": ["The Crown (show)", "Peaky Blinders (show)"],
    "period drama": ["Bridgerton (show)", "Downton Abbey (show)"],
    "jdrama": ["Alice in Borderland (show)", "Terrace House (show)"],
    "anime": ["Attack on Titan (show)", "Death Note (show)"],
    "webtoon": ["Solo Leveling (show)", "Tower of God (show)"],
    "webtoons": ["Solo Leveling (show)", "Tower of God (show)"],
    "manhwa": ["Solo Leveling (show)", "Tower of God (show)"],
    "manga": ["Death Note (show)", "One Piece (show)"],
    "romance": ["Pride and Prejudice (book)", "The Notebook (film)"],
    "fantasy": ["Harry Potter (book)", "Game of Thrones (show)"],
    "thriller": ["Gone Girl (film)", "Sherlock (show)"],
    "mystery": ["Sherlock (show)", "Knives Out (film)"],
    "murder": ["Only Murders in the Building (show)", "Knives Out (film)"],
    "crime": ["Breaking Bad (show)", "Peaky Blinders (show)"],
    "horror": ["The Conjuring (film)", "Stranger Things (show)"],
    "comedy": ["Friends (show)", "The Office (show)"],
    "sci fi": ["Interstellar (film)", "Dune (film)"],
    "scifi": ["Interstellar (film)", "Dune (film)"],
    "science fiction": ["Interstellar (film)", "Dune (film)"],
    "action": ["John Wick (film)", "The Dark Knight (film)"],
    "adventure": ["The Lord of the Rings (film)", "Indiana Jones (film)"],
}


# Book genres come from Qloo's own genre tags (GET /v2/tags, then insights with filter.tags; verified Oct 2026).
# A genre resolves to an Entity of this type whose id is its tag ids joined by commas.
GENRE_TYPE = "urn:tag:genre"
_TAG = "urn:tag:genre:media:"
# Reader words Qloo's tag search does not find, as the book tags that mean the same (several: a book must carry all).
# Some tags exist on books but never come back from GET /v2/tags ("dark_romance", "romantasy"): found on book results.
GENRE_TAGS = {
    "dark romance": ["dark_romance"],
    "romcom": ["contemporary_romance", "comedy"],
    "rom com": ["contemporary_romance", "comedy"],
    "romantic comedy": ["contemporary_romance", "comedy"],
    "romantasy": ["romantasy"],
    "spicy": ["erotic_romance"],
    "spicy romance": ["erotic_romance"],
    "smut": ["erotic_romance"],
    "ya": ["young_adult"],
    "sci fi": ["science_fiction"],
    "scifi": ["science_fiction"],
    "self help": ["self_help"],
}
# Words that make a short phrase look like a book genre ("cozy mystery", "dark fantasy"), so Auto-detect asks Qloo's
# tags before searching it as a title. Only a near-exact tag name is accepted that way; anything else stays a title.
_GENRE_WORDS = {"romance", "fantasy", "mystery", "thriller", "horror", "comedy", "fiction", "nonfiction", "crime",
                "lit", "poetry", "classics", "adventure", "paranormal", "gothic", "dystopian", "dystopia", "humor",
                "erotica", "contemporary", "manga", "manhwa", "webtoon", "webtoons", "suspense", "western", "memoir",
                "biography", "history", "historical", "action", "murder", "science fiction"}
# Screen genres keep their well-known stand-in shows (GENRE_ANCHORS): book tags do not describe them.
_SCREEN_GENRES = {"kdrama", "k drama", "korean drama", "cdrama", "c drama", "chinese drama", "historical cdrama",
                  "historical drama", "period drama", "jdrama", "anime", "drama", "dramas"}


def is_genre(entity: Entity) -> bool:
    return entity.type == GENRE_TYPE


def genre_tag_names(entity: Entity) -> str:
    """'urn:tag:genre:media:dark,urn:tag:genre:media:romance' -> 'Dark + Romance'."""
    return " + ".join(t.rsplit(":", 1)[-1].replace("_", " ").capitalize() for t in entity.id.split(","))


def _canon(base: str) -> str:
    """Normalised name, with another spelling of a trope turned into ours ('rivals to lovers' -> 'enemies to lovers')."""
    key = norm_title(base)
    return _TROPE_ALIASES.get(key, key)


def _tag_ids(key: str) -> Optional[list[str]]:
    """Full Qloo tag ids for a genre or trope we map by hand, or None."""
    if key in TROPE_TAGS:
        return ["urn:tag:" + TROPE_TAGS[key]]
    if key in GENRE_TAGS:
        return [_TAG + t for t in GENRE_TAGS[key]]
    return None


def is_trope(name: str) -> bool:
    base, hint = parse_hint(name)
    return hint == "trope" or _canon(base) in TROPE_TAGS or _canon(base) in _TROPES


def looks_like_genre(name: str) -> bool:
    """True when a typed favorite may be a book genre or tagged trope worth checking against Qloo's tags (no request)."""
    base, hint = parse_hint(name)
    key = _canon(base)
    if hint == "trope":
        return key in TROPE_TAGS
    if hint == "genre":
        return key not in _SCREEN_GENRES
    if hint is not None or key in _SCREEN_GENRES:
        return False
    words = key.split()
    return (key in GENRE_TAGS or key in TROPE_TAGS or key in _GENRE_WORDS
            or (len(words) <= 3 and words[-1] in _GENRE_WORDS))


# Story tropes Qloo tags on books. Its tag search only shows the show and film versions, but books carry these
# (seen on book results, Oct 2026), and filtering by each gave well-known books of that trope.
TROPE_TAGS = {
    "enemies to lovers": "genre:media:enemies_to_lovers",
    "friends to lovers": "genre:media:friends_to_lovers",
    "second chance": "genre:media:second_chance",
    "fake dating": "keyword:qloo:fake_dating",
    "love triangle": "keyword:qloo:love_triangle",
    "arranged marriage": "keyword:qloo:arranged_marriage",
    "billionaire": "genre:media:billionaire_romance",
    "small town": "genre:media:small_town_romance",
    "mafia romance": "genre:media:mafia_romance",
    "sports romance": "genre:media:sports_romance",
    "workplace romance": "genre:media:workplace_romance",
    "reverse harem": "genre:media:reverse_harem",
}
# Every trope also stands in as books famous for it (each checked to resolve to the right book). The taste graph works
# from those when a trope has no usable book tag (Qloo's tags for these return one or two obscure books, or none),
# and offline, when there are no tags at all.
_TROPES = {
    "enemies to lovers": ["The Cruel Prince (book)", "The Hating Game (book)"],
    "friends to lovers": ["People We Meet on Vacation (book)", "Love, Rosie (book)"],
    "fake dating": ["The Love Hypothesis (book)", "To All the Boys I've Loved Before (book)"],
    "grumpy sunshine": ["Book Lovers (book)", "Icebreaker (book)"],
    "forced proximity": ["The Unhoneymooners (book)", "The Flatshare (book)"],
    "second chance": ["Happy Place (book)", "Every Summer After (book)"],  # Persuasion pulled in Victorian classics
    "slow burn": ["Pride and Prejudice (book)", "The Spanish Love Deception (book)"],
    "forbidden love": ["Romeo and Juliet (book)", "Twilight (book)"],
    "found family": ["Six of Crows (book)", "The Long Way to a Small, Angry Planet (book)"],
    "love triangle": ["The Hunger Games (book)", "The Summer I Turned Pretty (book)"],
    "chosen one": ["Harry Potter and the Sorcerer's Stone (book)", "The Lightning Thief (book)"],
    "morally grey": ["Six of Crows (book)", "Vicious (book)"],
    "arranged marriage": ["King of Wrath (book)", "The Wrath and the Dawn (book)"],
    "brothers best friend": ["Twisted Love (book)", "The Deal (book)"],
    "billionaire": ["King of Wrath (book)", "Fifty Shades of Grey (book)"],
    "small town": ["Virgin River (book)", "Book Lovers (book)"],
    "secret relationship": ["Red, White & Royal Blue (book)", "Twisted Games (book)"],
    "fated mates": ["A Court of Thorns and Roses (book)", "From Blood and Ash (book)"],
    "revenge": ["The Count of Monte Cristo (book)", "Vicious (book)"],
}
# Other ways readers write the same trope.
_TROPE_ALIASES = {
    "enemies 2 lovers": "enemies to lovers", "rivals to lovers": "enemies to lovers", "hate to love": "enemies to lovers",
    "friends 2 lovers": "friends to lovers", "best friends to lovers": "friends to lovers",
    "fake relationship": "fake dating", "grumpy x sunshine": "grumpy sunshine", "grumpy and sunshine": "grumpy sunshine",
    "grumpy sunshine romance": "grumpy sunshine", "only one bed": "forced proximity", "one bed": "forced proximity",
    "second chance romance": "second chance", "second chances": "second chance", "slowburn": "slow burn",
    "forbidden romance": "forbidden love", "morally gray": "morally grey", "morally grey hero": "morally grey",
    "marriage of convenience": "arranged marriage", "brother s best friend": "brothers best friend",
    "billionaire romance": "billionaire", "small town romance": "small town", "secret dating": "secret relationship",
    "mates": "fated mates", "the chosen one": "chosen one", "office romance": "workplace romance",
    "boss romance": "workplace romance", "hockey romance": "sports romance", "why choose": "reverse harem",
}
TROPES = list(dict.fromkeys([*_TROPES, *TROPE_TAGS]))  # shown in the app's help


def trope_anchors(name: str) -> Optional[list[str]]:
    """Books famous for a trope the reader typed ('enemies to lovers', 'Enemies-to-lovers (trope)'), or None."""
    base, hint = parse_hint(name)
    if hint not in (None, "trope"):
        return None
    return _TROPES.get(_canon(base))


def genre_anchors(name: str) -> Optional[list[str]]:
    """Well-known titles standing in for a bare genre word, or None when `name` is a real title or has a type."""
    whole = GENRE_ANCHORS.get(norm_title(name))  # "historical cdrama" is a genre as a whole
    if whole and "(" not in name:
        return whole
    return GENRE_ANCHORS.get(norm_title(parse_hint(name)[0]))  # a type chosen next to a genre word changes nothing


def is_generic(name: str) -> bool:
    """'kdrama' or 'japan' describes a genre or place, not one title."""
    return norm_title(parse_hint(name)[0]) in _GENERIC



def norm_title(s: str) -> str:
    """Lowercase, drop punctuation and a leading 'the': 'GTA-5' and 'gta 5' compare equal."""
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower().replace("&", " and ")).strip()
    return re.sub(r"^(the|a|an)\s+", "", s)


def query_variants(name: str) -> list[str]:
    """Spellings worth searching, best guess first.

    'gta-5' -> ['Grand Theft Auto V', 'Grand Theft Auto 5', 'gta 5', 'gta-5'].
    Qloo search is literal, so this is what makes short forms and hyphens work.
    """
    raw = (name or "").strip()
    if not raw:
        return []
    spaced = re.sub(r"[-_.]+", " ", raw)
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced) if len(spaced) <= 12 else spaced
    tokens = spaced.split()
    expanded = " ".join(_ALIASES.get(t.lower(), t) for t in tokens)
    roman = " ".join(_ROMAN[int(t)] if t.isdigit() and int(t) in _ROMAN and i > 0 else t
                     for i, t in enumerate(expanded.split()))
    out: list[str] = []
    for v in (roman, expanded, " ".join(tokens), raw):
        v = " ".join(v.split())
        if v and v.lower() not in {o.lower() for o in out}:
            out.append(v)
    return out


def similarity(a: str, b: str) -> float:
    """0..1 closeness of two titles; containment ('Grand Theft Auto V' in 'Grand Theft Auto V: Premium') counts high.
    A series tag is ignored: 'Twisted Love' equals 'Twisted Love (Twisted, #1)'."""
    na, nb = norm_title(_SERIES_TAG.sub("", a or "")), norm_title(_SERIES_TAG.sub("", b or ""))
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    if na in nb or nb in na:
        ratio = max(ratio, 0.85 * min(len(na), len(nb)) / max(len(na), len(nb)) + 0.15)
    if nb.startswith(na + " ") or na.startswith(nb + " "):  # "avatar" vs "avatar the last airbender"
        ratio = max(ratio, 0.7)
    return ratio


_MOCK_LOOKUP = {norm_title(k): v for k, v in mock_data.MOCK_ENTITIES.items()}


class QlooUnavailable(Exception):
    """Raised when Qloo cannot be reached after retries (rate limit, outage) or refuses the key."""


class QlooRejected(Exception):
    """Qloo answered but refused this one request (HTTP 400/404/422), for example an entity it cannot use as
    a signal. The key and the service are fine, so callers skip that item instead of giving up on Qloo."""


class QlooClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://hackathon.api.qloo.com",
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        min_interval: float = 0.21,  # Qloo allows 5 requests per second
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._clock = clock
        self.min_interval = min_interval
        self._last_request: Optional[float] = None
        self._lock = threading.Lock()  # the throttle is shared by threads that fetch several favorites at once
        self._cache: dict[tuple, tuple[float, object]] = {}
        self.calls_made = 0  # real HTTP requests, for tests and the trace
        self.bad_ids: set[str] = set()  # entities Qloo refused to use as a signal; resolve() skips them next time

    @property
    def is_mock(self) -> bool:
        return not self.api_key

    # ------------------------------------------------------------------ public

    def resolve(self, name: str) -> Optional[Entity]:
        """Resolve a free-text favorite to an entity, or None if unrecognized."""
        base, hint = parse_hint(name)
        if base and looks_like_genre(name):
            genre = self.resolve_genre(name)
            if genre is not None:
                return genre
        if not base or is_generic(name) or hint in ("genre", "trope"):  # an unknown genre or trope is not a title
            return None
        if self.is_mock:
            hit = next((h for h in (_MOCK_LOOKUP.get(norm_title(v)) for v in query_variants(base)) if h), None)
            if not hit:
                return None
            display, etype = hit
            return Entity(id=f"mock:{display.lower()}", name=display, type=etype, source="mock")
        best: Optional[dict] = None
        best_sim = 0.0
        variants = [*known_alias(base, hint), *query_variants(base)]
        for i, q in enumerate(variants):
            results: list[dict] = []
            try:
                if hint:  # Qloo's search accepts a `types` filter (verified in the Oct 2026 probe)
                    results = _extract_entities(
                        self._get("/search", {"query": q, "take": SEARCH_TAKE, "types": f"urn:entity:{hint}"})
                    )
                if not results:
                    results = _extract_entities(self._get("/search", {"query": q, "take": SEARCH_TAKE}))
            except QlooRejected:
                continue
            results = [r for r in results if str(r.get("entity_id") or r.get("id")) not in self.bad_ids]
            usable = [r for r in results if kind_of(_entity_type(r)) not in _UNUSABLE_TYPES]
            e = pick_entity(usable or results, variants, hint)
            if e is None:
                continue
            sim = max(similarity(v, str(e.get("name", ""))) for v in variants)
            if sim > best_sim:
                best, best_sim = e, sim
            if sim >= 0.9 or i >= 2:  # a clear match, or enough tries (each is one request)
                break
        if not hint and best is not None and kind_of(_entity_type(best)) in _UNUSABLE_TYPES:
            # Only a place/album matched ("babrie in swan lake" -> "Swan Lake Resort"). Look for a real title first.
            better = self._typed_retry(variants)
            if better is not None and better[1] >= max(MIN_SIMILARITY, best_sim - 0.05):
                best, best_sim = better
        if best is None or best_sim < MIN_SIMILARITY:
            return None
        e = best
        return Entity(id=str(e.get("entity_id") or e.get("id")), name=html.unescape(str(e.get("name") or base)),
                      type=_entity_type(e), source="qloo",
                      author=_author(e) if kind_of(_entity_type(e)) == "book" else "")

    def resolve_genre(self, name: str) -> Optional[Entity]:
        """A book genre as Qloo genre tags, or None. Chosen as a genre, the closest tag is fine ('mafia thriller'
        -> 'Thriller', flagged as uncertain); on Auto-detect only a near-exact tag name counts."""
        base, hint = parse_hint(name)
        if self.is_mock or not looks_like_genre(name):
            return None
        ids = _tag_ids(_canon(base))
        if ids:
            return Entity(id=",".join(ids), name=base, type=GENRE_TYPE, source="qloo")
        if hint == "trope":  # tropes are only the ones we know; Qloo's tag search would match show tags
            return None
        try:
            data = self._get("/v2/tags", {"filter.query": base, "filter.parents.types": "urn:entity:book", "take": 5})
        except QlooRejected:
            return None
        tags = data.get("results", {}).get("tags", []) if isinstance(data, dict) else []
        tags = [t for t in tags if isinstance(t, dict) and str(t.get("id", "")).startswith(_TAG)]
        if not tags:
            return None
        best = max(tags, key=lambda t: similarity(base, str(t.get("name", ""))))
        if similarity(base, str(best.get("name", ""))) < (MIN_SIMILARITY if hint == "genre" else 0.9):
            return None
        return Entity(id=str(best["id"]), name=str(best.get("name") or base), type=GENRE_TYPE, source="qloo")

    def _typed_retry(self, variants: list[str]) -> Optional[tuple[dict, float]]:
        """Search the kinds of title a reader usually means, one type at a time. Returns (entity, similarity)."""
        best: Optional[tuple[dict, float]] = None
        for kind in ("movie", "tv_show", "videogame", "artist", "book"):
            try:
                results = _extract_entities(
                    self._get("/search", {"query": variants[0], "take": SEARCH_TAKE, "types": f"urn:entity:{kind}"})
                )
            except QlooRejected:
                continue
            results = [r for r in results if str(r.get("entity_id") or r.get("id")) not in self.bad_ids]
            e = pick_entity(results, variants, kind)
            if e is None:
                continue
            sim = max(similarity(v, str(e.get("name", ""))) for v in variants)
            if best is None or sim > best[1]:
                best = (e, sim)
            if sim >= 0.9:
                break
        return best

    def book_recs(self, seeds: list[Entity], per_seed: int = 8) -> list[Candidate]:
        """Books the taste graph links to the seeds, merged and ranked.

        One call per seed, so every book keeps the list of favorites that led to
        it. That linkage is the evidence shown to the user.
        """
        pool: dict[str, Candidate] = {}
        for seed in seeds:
            merge_into(pool, self.recs_for_seed(seed, per_seed))
        return rank(pool)

    def recs_for_seed(self, seed: Entity, take: int = 8) -> list[Candidate]:
        """Books the taste graph links to ONE favorite (one Qloo call, cached)."""
        if self.is_mock:
            rows = mock_data.MOCK_BOOK_RECS.get(seed.name, [])
            return [
                Candidate(
                    id=f"mock:{title.lower()}",
                    name=title,
                    author=author,
                    tags=list(tags),
                    affinity=aff,
                    seeds=[seed.name],
                    pages=pages,
                    source="mock",
                )
                for title, author, tags, aff, pages in rows[:take]
            ]
        if is_genre(seed):  # the best-known books carrying every one of the genre's tags
            params = {"filter.type": "urn:entity:book", "filter.tags": seed.id,
                      "operator.filter.tags": "intersection", "take": take}
        else:
            params = {
                "filter.type": "urn:entity:book",
                "signal.interests.entities": seed.id,
                "take": take,
            }
        return self._parse_books(self._get("/v2/insights", params), [seed.name])

    def recs_for_group(self, seeds: list[Entity], take: int = 10) -> list[Candidate]:
        """Books for the WHOLE mix of favorites at once (one Qloo call, cached).

        Qloo takes several entities as one taste signal, so the answer fits the
        combination rather than any single favorite. `blend` records which
        favorites were in that signal.
        """
        names = [s.name for s in seeds]
        if len(seeds) < 2:
            return []
        if self.is_mock:
            rows = []
            for seed in seeds:
                rows += self.recs_for_seed(seed, 4)
            best = {}
            for c in rows:
                if c.id not in best or (c.affinity or 0) > (best[c.id].affinity or 0):
                    best[c.id] = c
            top = sorted(best.values(), key=lambda c: -(c.affinity or 0))[:take]
            return [Candidate(id=c.id, name=c.name, author=c.author, tags=list(c.tags), affinity=c.affinity,
                              seeds=[], blend=names, pages=c.pages, source="mock") for c in top]
        # A genre is not a taste signal but a filter: "Fourth Wing" + "dark romance" gives dark romance that Fourth
        # Wing readers like. With two genres there is no single filter, so only the titles are combined.
        titles = [s for s in seeds if not is_genre(s)]
        genres = [s for s in seeds if is_genre(s)]
        if not titles or (len(titles) < 2 and len(genres) != 1):
            return []
        params = {
            "filter.type": "urn:entity:book",
            "signal.interests.entities": ",".join(s.id for s in titles),
            "take": take,
        }
        if len(genres) == 1:
            params.update({"filter.tags": genres[0].id, "operator.filter.tags": "intersection"})
        else:
            names = [s.name for s in titles]
        out = self._parse_books(self._get("/v2/insights", params), seeds=[])
        for c in out:
            c.blend = names
        return out

    def _parse_books(self, data: object, seeds: list[str]) -> list[Candidate]:
        out: list[Candidate] = []
        for e in _extract_entities(data):
            eid = e.get("entity_id") or e.get("id")
            name = html.unescape(str(e.get("name") or "Unknown title"))  # Qloo sends some as "Red, White &amp; Royal Blue"
            # Box sets, colouring books and guides are a poor way back into reading.
            if not eid or _OMNIBUS.search(name):
                continue
            props = e.get("properties") or {}
            rating, count = _goodreads(e)
            out.append(
                Candidate(
                    id=str(eid), name=name, author=_author(e), tags=_tag_names(e), affinity=_affinity(e),
                    seeds=list(seeds), pages=_int(props.get("page_count")), source="qloo",
                    image_url=_image_url(e), year=_int(props.get("publication_year")),
                    rating=rating, rating_count=count,
                )
            )
        return out

    def _throttle(self) -> None:
        """Stay under Qloo's 5 requests/second limit instead of earning 429s."""
        with self._lock:
            if self.min_interval and self._last_request is not None:
                wait = self.min_interval - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()

    def _get(self, path: str, params: dict) -> object:
        key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
        now = self._clock()
        cached = self._cache.get(key)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

        url = f"{self.base_url}{path}"
        headers = {"X-Api-Key": self.api_key}
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=20)
                self.calls_made += 1
            except requests.RequestException as exc:  # network problem
                last_error = f"network error: {exc.__class__.__name__}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    self._cache[key] = (now, data)
                    return data
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                else:
                    # 4xx other than 429 will not get better with retries.
                    detail = f"HTTP {resp.status_code} from {path}{_error_hint(resp)}"
                    if resp.status_code in (400, 404, 422):
                        raise QlooRejected(detail)
                    raise QlooUnavailable(detail)  # 401/403: the key itself is the problem
            if attempt < self.max_retries:
                self._sleep(self.backoff_seconds * (2**attempt))
        raise QlooUnavailable(f"{path} failed after retries ({last_error})")


def _error_hint(resp: object) -> str:
    """A short piece of Qloo's own error text, so a 400 says WHY. Never contains the key (it is only sent as a header)."""
    text = str(getattr(resp, "text", "") or "").strip().replace("\n", " ")
    return f": {text[:200]}" if text else ""


def parse_hint(name: str) -> tuple[str, Optional[str]]:
    """'Hades (game)' -> ('Hades', 'videogame'). Unknown hints stay part of the name."""
    name = (name or "").strip()
    m = _HINT_RE.match(name)
    if m and m.group(2).strip().lower() in _HINTS:
        return m.group(1).strip(), _HINTS[m.group(2).strip().lower()]
    t = _TRAIL_RE.match(name)
    if t:
        return t.group(1).strip(), _TRAIL_KIND[re.sub(r"[\s-]+", "", t.group(2).lower())]
    return name, None


def kind_of(entity_type: str) -> str:
    """'urn:entity:videogame' -> 'videogame'."""
    return (entity_type or "").rsplit(":", 1)[-1] or "unknown"


def pick_entity(results: list[dict], name, hint: Optional[str]) -> Optional[dict]:
    """Choose the search result the reader most likely meant.

    `name` is one spelling or a list of spellings. Order: results of the hinted
    type (if any), then closest title, then not far less popular than the best
    equally close title, then the type priority above, then Qloo's popularity.
    Returns None when no result has an id.
    """
    usable = [r for r in results if r.get("entity_id") or r.get("id")]
    if not usable:
        return None
    if hint:
        hinted = [r for r in usable if kind_of(_entity_type(r)) == hint]
        usable = hinted or usable
    names = [name] if isinstance(name, str) else list(name)

    def closeness(r: dict) -> tuple:
        sim = max((similarity(n, str(r.get("name", ""))) for n in names), default=0.0)
        return (0 if sim >= 0.999 else 1, -round(sim, 1))

    def popularity(r: dict) -> float:
        pop = r.get("popularity")
        return pop if isinstance(pop, (int, float)) else 0.0

    top_pop: dict[tuple, float] = {}  # most popular result among each set of equally close titles
    for r in usable:
        c = closeness(r)
        top_pop[c] = max(top_pop.get(c, 0.0), popularity(r))

    def score(r: dict) -> tuple:
        kind = kind_of(_entity_type(r))
        prio = _TYPE_PRIORITY.index(kind) if kind in _TYPE_PRIORITY else len(_TYPE_PRIORITY)
        c, pop = closeness(r), popularity(r)
        return (*c, pop < top_pop[c] - POPULARITY_GAP, prio, -pop)

    return sorted(usable, key=score)[0]


def merge_into(pool: dict[str, Candidate], cands: list[Candidate]) -> None:
    """Add candidates to a pool, keeping every favorite that led to a book.

    The same book under different ids (editions) is merged by normalized title.
    """
    for cand in cands:
        existing = pool.get(cand.id)
        if existing is None:
            key = norm_title(cand.name)
            existing = next((c for c in pool.values() if norm_title(c.name) == key), None)
        if existing is None:
            pool[cand.id] = cand
            continue
        existing.seeds = sorted(set(existing.seeds) | set(cand.seeds))
        existing.blend = sorted(set(existing.blend) | set(cand.blend))
        existing.tags = sorted(set(existing.tags) | set(cand.tags))
        if cand.affinity is not None:
            prev = existing.affinity if existing.affinity is not None else cand.affinity
            existing.affinity = max(prev, cand.affinity)


def rank(pool: dict[str, Candidate]) -> list[Candidate]:
    """Books that fit more of the reader's favorites come first (own lists or the combined query), then affinity."""
    return sorted(pool.values(), key=lambda c: (-len(set(c.seeds) | set(c.blend)), -(c.affinity or 0.0), c.name))


# ---------------------------------------------------------------- parse helpers
# Defensive on purpose: the real response shape is unverified.


def _extract_entities(data: object) -> list[dict]:
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        for key in ("results", "entities", "data"):
            if key in data:
                inner = _extract_entities(data[key])
                if inner:
                    return inner
    return []


def _entity_type(e: dict) -> str:
    """Search uses `types` (a list); insights uses `subtype` ('urn:entity:book')."""
    if e.get("subtype"):
        return str(e["subtype"])
    types = e.get("types") or e.get("type") or []
    if isinstance(types, list):
        return str(types[0]) if types else ""
    return str(types)


def _tag_names(e: dict) -> list[str]:
    """Genre and theme tags. Qloo's audience labels ('Dedicated', 'Loyal') are noise here."""
    names = []
    for t in e.get("tags") or []:
        if isinstance(t, dict) and t.get("name"):
            if ":audience" in str(t.get("type", "")) or ":audience" in str(t.get("tag_id", "")):
                continue
            names.append(str(t["name"]))
        elif isinstance(t, str):
            names.append(t)
    return names


def _author(e: dict) -> str:
    """Qloo gives the author inside `disambiguation`, e.g. '2018, Madeline Miller'."""
    props = e.get("properties") or {}
    value = props.get("author") or props.get("authors")
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    if value:
        return html.unescape(str(value))
    parts = html.unescape(str(e.get("disambiguation") or "")).split(",", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1].strip()
    return ""


def _int(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _image_url(e: dict) -> str:
    url = ((e.get("properties") or {}).get("image") or {}).get("url") or ""
    return url if isinstance(url, str) and url.startswith("https://images.qloo.com/") else ""


def _goodreads(e: dict) -> tuple[Optional[float], Optional[int]]:
    """Goodreads data is a list on insights and a dict on search; take either."""
    gr = (e.get("external") or {}).get("goodreads") or (e.get("properties", {}).get("external") or {}).get("goodreads")
    if isinstance(gr, list):
        gr = gr[0] if gr else None
    if not isinstance(gr, dict):
        return None, None
    try:
        rating = float(gr["user_rating"]) if gr.get("user_rating") is not None else None
    except (TypeError, ValueError):
        rating = None
    return rating, _int(gr.get("user_ratings_count"))


def _affinity(e: dict) -> Optional[float]:
    q = e.get("query") or {}
    value = q.get("affinity") if isinstance(q, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
