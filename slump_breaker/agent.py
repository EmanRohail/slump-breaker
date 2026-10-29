"""The Slump Breaker agent: a short, bounded tool-calling loop.

The model (Gemini, or a fixed-plan mock offline) decides which tool to call:

    resolve_entity(name)        find a favorite in the taste graph
    get_book_recs(entity_name)  books the graph links to that favorite
    finish(picks)               the final three, with a one-line reason each

Code, not the model, runs the tools, filters out books the reader rejected, and
checks that every final pick really came back from the taste graph. Every step
lands in a trace the UI can show.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from .llm import GeminiLLM, LLMError, MockLLM, _template_why
from .models import AgentResult, Candidate, Entity, Pick, ToolCall, TraceStep, spoken
from .qloo import (QlooClient, QlooRejected, QlooUnavailable, genre_anchors, genre_tag_names, is_generic, kind_of,
                   is_genre, is_trope, known_alias, looks_like_genre, merge_into, norm_title, parse_hint, query_variants, rank,
                   trope_anchors)
from .slump_types import get_slump

MAX_FAVORITES = 5
MAX_TURNS = 6  # tool rounds before we stop waiting for the model


def clean_favorites(raw: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        name = (item or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out[:MAX_FAVORITES]  # the UI also stops people at this limit


@dataclass
class _Run:
    """Everything one attempt of the loop learns."""

    client: QlooClient
    exclude_ids: set[str]
    exclude_names: set[str]
    entities: dict[str, Entity] = field(default_factory=dict)  # lowercase display name -> entity
    typed: dict[str, str] = field(default_factory=dict)  # lowercase display name -> what the reader typed
    unresolved: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    pool: dict[str, Candidate] = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    final: Optional[list[dict]] = None
    trope_books: set[str] = field(default_factory=set)  # stand-in titles typed for a trope ("The Cruel Prince (book)")
    avoid_authors: set[str] = field(default_factory=set)  # their authors: a trope wants other writers, not sequels
    held: dict[str, Candidate] = field(default_factory=dict)  # books by those authors, used only to reach three picks


def _keep(run: _Run, found: list[Candidate]) -> tuple[list[Candidate], int]:
    """Books not rejected by the reader and, for a trope, not by its example books' authors (those are held back).
    Returns the books and how many were held back."""
    cands = [c for c in found if c.id not in run.exclude_ids and c.name.lower() not in run.exclude_names]
    others = [c for c in cands if c.author.lower() not in run.avoid_authors]
    merge_into(run.held, [c for c in cands if c.author.lower() in run.avoid_authors])
    return others, len(cands) - len(others)


def _book_view(c: Candidate) -> dict:
    view = {
        "id": c.id,
        "title": c.name,
        "author": c.author,
        "tags": c.tags[:6],
        "affinity": c.affinity,
        "linked_favorites": c.seeds,
    }
    if c.blend:
        view["fits_combined_taste_of"] = c.blend
    if c.pages:
        view["pages"] = c.pages
    return view


def _seed_books(run: _Run, entity: Entity) -> list[Candidate]:
    """Books for one favorite. If Qloo refuses the matched entity as an input, try the next-best match once."""
    try:
        return run.client.recs_for_seed(entity)
    except QlooRejected:
        run.client.bad_ids.add(entity.id)
        typed = run.typed.get(entity.name.lower(), entity.name)
        again = run.client.resolve(typed)
        if again is None or again.id == entity.id:
            raise
        if again.name.lower() != entity.name.lower():
            run.entities.pop(entity.name.lower(), None)
        run.entities[again.name.lower()] = again
        run.typed[again.name.lower()] = typed
        run.trace.append(TraceStep("resolve_entity", f"'{entity.name}' was refused as an input; retrying with {again.name}"))
        return run.client.recs_for_seed(again)


def _execute(call: ToolCall, run: _Run) -> dict:
    if call.name == "resolve_entity":
        name = str(call.args.get("name", "")).strip()
        entity = run.client.resolve(name)
        if entity is None:
            run.unresolved.append(name)
            why = "is a genre, topic or place, not a title; try a specific one" if is_generic(name) else "no match in the taste graph"
            run.trace.append(TraceStep("resolve_entity", f"'{name}': {why}", ok=False))
            return {"found": False, "name": name}
        run.entities[entity.name.lower()] = entity
        run.typed[entity.name.lower()] = name
        if name in run.trope_books and entity.author:
            run.avoid_authors.add(entity.author.lower())
        kind = kind_of(entity.type)
        if is_genre(entity):
            run.matches.append(f"{parse_hint(name)[0]} → Qloo tags {genre_tag_names(entity)}")
        else:
            run.matches.append(f"{name} → {entity.name} ({kind})")
        run.trace.append(TraceStep("resolve_entity", f"'{name}' -> {entity.name} ({kind})"))
        return {"found": True, "entity": {"name": entity.name, "type": entity.type}}

    if call.name == "get_book_recs":
        name = str(call.args.get("entity_name", "")).strip()
        entity = run.entities.get(name.lower())
        if entity is None:
            run.trace.append(TraceStep("get_book_recs", f"'{name}' was not resolved first", ok=False))
            return {"error": "unknown entity: call resolve_entity first and use the returned name"}
        try:
            found = _seed_books(run, entity)
        except QlooRejected as exc:
            run.entities.pop(entity.name.lower(), None)
            run.unresolved.append(run.typed.get(entity.name.lower(), name))
            run.trace.append(TraceStep("get_book_recs", f"Qloo cannot use '{entity.name}' as an input ({exc}); skipped", ok=False))
            return {"error": f"'{name}' cannot be used; ignore this favorite"}
        cands, skipped = _keep(run, found)
        merge_into(run.pool, cands)
        run.trace.append(TraceStep("get_book_recs", f"{len(cands)} books linked to {entity.name}"
                                   + (f" ({skipped} by the trope's example authors left out)" if skipped else "")))
        return {"entity": entity.name, "books": [_book_view(c) for c in cands]}

    if call.name == "get_blended_recs":
        ents = list(run.entities.values())
        if len(ents) < 2:
            run.trace.append(TraceStep("get_blended_recs", "needs two or more resolved favorites", ok=False))
            return {"error": "need at least two resolved favorites"}
        try:
            group = run.client.recs_for_group(ents)
        except (QlooUnavailable, QlooRejected) as exc:
            # The combined query is a bonus. If Qloo refuses it, keep the per-favorite results we already have.
            run.trace.append(TraceStep("get_blended_recs", f"combined query failed ({exc}); using each favorite's own list", ok=False))
            return {"error": "combined recommendations unavailable; choose from get_book_recs results"}
        if not group:  # e.g. only genres: they filter books, they are not a taste to combine
            run.trace.append(TraceStep("get_blended_recs", "these favorites cannot be combined; using each one's own list", ok=False))
            return {"error": "combined recommendations unavailable; choose from get_book_recs results"}
        cands, skipped = _keep(run, group)
        merge_into(run.pool, cands)
        run.trace.append(TraceStep("get_blended_recs", f"{len(cands)} books fit all {len(ents)} favorites together"
                                   + (f" ({skipped} by the trope's example authors left out)" if skipped else "")))
        return {"favorites": [e.name for e in ents], "books": [_book_view(c) for c in cands]}

    run.trace.append(TraceStep(call.name, "unknown tool requested", ok=False))
    return {"error": f"unknown tool {call.name!r}"}


def _same_title(typed: str, matched: str) -> bool:
    """True when the match is the title that was typed (spacing, case, 'gta 5' short forms ignored)."""
    squash = lambda t: norm_title(t).replace(" ", "")  # noqa: E731
    return any(squash(v) == squash(matched) for v in [*query_variants(typed), *known_alias(typed, None)])


def _prefetch(run: _Run, calls: list[ToolCall]) -> None:
    """Fetch what this round of tool calls needs, several requests at a time. The results land in the client's
    cache, so the calls that follow (run in order, as before) answer instantly. Failures are ignored here and
    handled by the normal path."""
    if run.client.is_mock:
        return
    jobs = []
    for c in calls:
        if c.name == "resolve_entity":
            jobs.append(lambda n=str(c.args.get("name", "")).strip(): run.client.resolve(n))
        elif c.name == "get_book_recs":
            e = run.entities.get(str(c.args.get("entity_name", "")).strip().lower())
            if e is not None:
                jobs.append(lambda e=e: run.client.recs_for_seed(e))
        elif c.name == "get_blended_recs" and len(run.entities) >= 2:
            jobs.append(lambda ents=list(run.entities.values()): run.client.recs_for_group(ents))
    if len(jobs) < 2:
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in [pool.submit(j) for j in jobs]:
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - the sequential pass reports it properly
                pass


def _loop(client: QlooClient, llm, task: dict, exclude_ids: set[str], exclude_names: set[str],
          trope_books: Optional[set[str]] = None) -> _Run:
    run = _Run(client=client, exclude_ids=exclude_ids, exclude_names=exclude_names, trope_books=set(trope_books or ()))
    session = llm.new_session(task)
    results: Optional[list] = None
    for _ in range(MAX_TURNS):
        calls = session.next_calls(results, rank(run.pool))
        results = []
        _prefetch(run, [c for c in calls if c.name != "finish"])
        for call in calls:
            if call.name == "finish":
                picks = call.args.get("picks")
                run.final = [p for p in picks if isinstance(p, dict)] if isinstance(picks, list) else []
                break
            results.append((call, _execute(call, run)))
        if run.final is not None:
            break
    if run.final is None:
        run.trace.append(TraceStep("finish", f"model did not finish in {MAX_TURNS} rounds; using top ranked books", ok=False))
        run.final = []
    return run


MIN_SWAP_RATINGS = 50_000  # a swapped-in book must be this well known: some favorites' lists are weak all the way down
                           # (Taylor Swift's are business exposés with ~1,500 ratings; the Dune film's are One Piece volumes)


def _known(c: Candidate) -> bool:
    return c.rating_count is None or c.rating_count >= MIN_SWAP_RATINGS  # no rating data (offline): allowed


MIN_PICK_RATINGS = 10_000  # below this a pick is little known (seen live: Sekiro Side Story 274 ratings, One Piece
                           # Vol. 22 8,849). Someone in a slump does better with a well-loved book.


def _lift(picks: list[Pick], ranked: list[Candidate]) -> tuple[list[Pick], list[str]]:
    """Replace each little-known pick with the best-ranked well-known book from the same taste-graph results,
    preferring one that links to a favorite the other picks do not cover. A pick stays when there is no such book,
    and books without rating data (offline) are never judged. Returns the picks and the titles replaced."""
    used = {p.candidate.id for p in picks}
    out, lifted = list(picks), []
    for i, p in enumerate(out):
        count = p.candidate.rating_count
        if count is None or count >= MIN_PICK_RATINGS:
            continue
        others = {s for j, q in enumerate(out) if j != i for s in q.candidate.seeds}
        known = [c for c in ranked if c.id not in used and (c.rating_count or 0) >= MIN_PICK_RATINGS]
        better = next((c for c in known if set(c.seeds) - others), None) or next(iter(known), None)
        if better is None:
            continue
        used.add(better.id)
        lifted.append(p.candidate.name.split(" (")[0])
        out[i] = Pick(candidate=better, why="")
    return out, lifted


def _spread(picks: list[Pick], ranked: list[Candidate]) -> tuple[list[Pick], int]:
    """Keep the first pick; swap a later one that adds no new favorite for the best-ranked book from a favorite no
    pick links to yet. The combined list favours one favorite's books (a book on both lists ranks first), which
    left every pick 'Linked to: Grand Theft Auto V' for a reader who also loves Stranger Things and Taylor Swift.
    Only well-known books are swapped in. Returns the picks and how many were swapped (their `why` is emptied)."""
    if len(picks) < 2:
        return picks, 0
    out = [picks[0]]
    covered = set(picks[0].candidate.seeds)
    used = {p.candidate.id for p in picks}
    swapped = 0
    for p in picks[1:]:
        if set(p.candidate.seeds) - covered:
            out.append(p)
            covered |= set(p.candidate.seeds)
            continue
        other = next((c for c in ranked if c.id not in used and set(c.seeds) - covered and _known(c)), None)
        if other is None:
            out.append(p)
            continue
        used.add(other.id)
        covered |= set(other.seeds)
        out.append(Pick(candidate=other, why=""))
        swapped += 1
    return out, swapped


def _book_genre(client: QlooClient, name: str) -> Optional[Entity]:
    """The favorite as Qloo genre tags, or None (not a genre, mock mode, or Qloo unreachable: stand-ins are used)."""
    if client.is_mock or not looks_like_genre(name):
        return None
    try:
        return client.resolve_genre(name)
    except QlooUnavailable:
        return None


def run(
    favorites: Iterable[str],
    reason_key: str,
    minutes: Optional[int],
    qloo: QlooClient,
    llm,
    exclude_ids: Optional[set[str]] = None,
    exclude_names: Optional[set[str]] = None,
) -> AgentResult:
    slump = get_slump(reason_key)
    favs = clean_favorites(favorites)
    genre_notes: list[TraceStep] = []
    anchor_genre: dict[str, str] = {}
    trope_books: set[str] = set()
    expanded: list[str] = []
    for f in favs:
        genre = _book_genre(qloo, f)
        if genre is not None:  # Qloo's own genre tags beat stand-in titles; resolve_entity finds it again (cached)
            what = "trope" if is_trope(f) else "genre"
            genre_notes.append(TraceStep("genre_tags", f"'{parse_hint(f)[0]}' is a {what}, so we used books Qloo tags as "
                                         + genre_tag_names(genre)))
            expanded.append(f)
            continue
        trope = trope_anchors(f)
        anchors = trope or genre_anchors(f)
        if anchors:
            what = "trope" if trope else "genre"
            genre_notes.append(TraceStep(f"{what}_anchor", f"'{parse_hint(f)[0]}' is a {what}, so we used well-known "
                                         f"{'books' if trope else 'titles'} from it: "
                                         + ", ".join(a.rsplit(" (", 1)[0] for a in anchors)))
            expanded += anchors
            trope_books.update(trope or [])
            anchor_genre.update({a: parse_hint(f)[0] for a in anchors})  # the reader's words, without "(trope)"
        else:
            expanded.append(f)
    favs = list(dict.fromkeys(expanded))
    ex_ids = set(exclude_ids or set())
    ex_names = {n.lower() for n in (exclude_names or set())}
    task = {"favorites": favs, "slump": slump, "minutes": minutes}
    trace: list[TraceStep] = [
        TraceStep("diagnose_slump", f"{slump.label}" + (" (prefers shorter reads)" if slump.prefers_short else "")),
        *genre_notes,
    ]

    client = qloo
    llm_used = llm
    mode = "mock" if qloo.is_mock else "live"
    attempt: Optional[_Run] = None
    for _ in range(3):  # at most one Qloo fallback and one model fallback
        try:
            attempt = _loop(client, llm_used, task, ex_ids, ex_names, trope_books)
            break
        except QlooUnavailable as exc:
            if client.is_mock:
                raise
            trace.append(TraceStep("taste_graph", f"unavailable ({exc}); using clearly labeled mock data", ok=False))
            client = QlooClient(api_key="")
            mode = "live-fallback-mock"
        except LLMError as exc:
            if isinstance(llm_used, MockLLM):
                raise
            trace.append(TraceStep("agent_model", f"{llm_used.mode} model failed ({exc}); using the fixed plan", ok=False))
            llm_used = MockLLM()
    assert attempt is not None
    trace.extend(attempt.trace)

    if ex_ids or ex_names:
        trace.append(TraceStep("exclusions", f"{len(attempt.pool)} candidates left after your feedback"))

    genre_of = {e.name: anchor_genre[attempt.typed[e.name.lower()]] for e in attempt.entities.values()
                if attempt.typed.get(e.name.lower()) in anchor_genre}
    matches = []
    for m in attempt.matches:  # "kdrama (via Squid Game) -> Squid Game (tv_show)" instead of the internal stand-in text
        typed, _, rest = m.partition(" → ")
        matches.append(f"{anchor_genre[typed]} (via {typed.rsplit(' (', 1)[0]}) → {rest}" if typed in anchor_genre else m)
    def said(c: Candidate) -> Candidate:  # the reader's own words in any sentence we write
        return replace(c, seeds=spoken(c.seeds, genre_of), blend=spoken(c.blend, genre_of))

    uncertain = []
    for e in attempt.entities.values():
        typed = attempt.typed.get(e.name.lower(), "")
        if typed and typed not in anchor_genre and not _same_title(parse_hint(typed)[0], e.name):
            uncertain.append(f"{typed} → {e.name}")
            genre_of[e.name] = parse_hint(typed)[0]  # any sentence says what was typed, not a guess at what was meant

    ranked = rank(attempt.pool)
    if len(ranked) < 3 and attempt.held:  # too few books by other writers: the trope's own authors fill the gap
        ranked += [c for c in rank(attempt.held) if c.id not in attempt.pool][:3 - len(ranked)]
        trace.append(TraceStep("trope_authors", "too few books by other writers, so some by the example books' authors were kept"))
    picks: list[Pick] = []
    if ranked:
        used: set[str] = set()
        dropped = 0
        for p in attempt.final or []:
            cand = attempt.pool.get(str(p.get("id", "")))
            if cand is None or cand.id in used:
                dropped += 1
                continue
            used.add(cand.id)
            why = str(p.get("why") or "")
            for stand_in, genre in genre_of.items():
                why = why.replace(stand_in, genre)
            picks.append(Pick(candidate=cand, why=why or _template_why(said(cand))))
        if dropped:
            trace.append(TraceStep("grounding_check", f"dropped {dropped} pick(s) not found in taste-graph results", ok=False))
        else:
            trace.append(TraceStep("grounding_check", "all picks come from taste-graph results"))
        for cand in ranked:  # top up if the model returned fewer than 3 valid picks
            if len(picks) >= 3:
                break
            if cand.id not in used:
                used.add(cand.id)
                picks.append(Pick(candidate=cand, why=_template_why(said(cand))))
        picks, swapped = _spread(picks[:3], ranked)
        if swapped:
            trace.append(TraceStep("spread_picks", f"swapped {swapped} pick(s) so more of the favorites lead to a book"))
        picks, lifted = _lift(picks, ranked)
        if lifted:
            trace.append(TraceStep("known_picks", f"replaced {len(lifted)} little-known pick(s) ("
                                   + "; ".join(lifted) + ") with better-known books from the same results"))
        picks = [p if p.why else Pick(candidate=p.candidate, why=_template_why(said(p.candidate))) for p in picks]

    return AgentResult(
        slump_key=slump.key,
        slump_label=slump.label,
        way_back=slump.way_back,
        picks=picks[:3],
        unresolved=attempt.unresolved,
        trace=trace,
        mode=mode,
        llm_mode=llm_used.mode,
        matches=matches,
        genre_of=genre_of,
        uncertain=uncertain,
        resolved=[e.name for e in attempt.entities.values()],
    )


def build_llm(gemini_key: str, model: str):
    return GeminiLLM(gemini_key, model) if gemini_key else MockLLM()
