# Slump Breaker

Win back lapsed readers with picks grounded in [Qloo](https://qloo.com)'s taste graph. Built for the Qloo Agentic Hackathon.

A book business (reading app, book box, publisher, library) gives Slump Breaker the readers it already knows have stopped coming back: a name, a few favorites and, optionally, why they stopped. For each reader the agent asks Qloo which books connect to those favorites, picks a comeback book, and writes a personal message for the channel the business uses (email, push or SMS).

A favorite can be a TV show or series, movie, video game, music artist or band, book, **book genre** (dark romance, romcom, cozy mystery…) or **trope** (enemies to lovers, fake dating, found family…). Each is entered in its own box with a type, or left on Auto-detect. In a CSV, add the same hint in brackets, e.g. `Healer (show); BTS (artist); slow burn (trope)`. For an album or single song, use the artist.

A **Reader** view shows the same engine from the reader's side, with "Worked / Not for me" feedback that re-asks Qloo.

**Live demo:** https://slump-breaker-xgmsapphsectx3mcq2nkr8z.streamlit.app/  (free hosting: if it says the app is asleep, click to wake it; it takes about a minute)

## Why it needs Qloo

A plain LLM recommends popular books and has no link between *this* reader's favorites and the books it suggests. Here the candidate books come from Qloo's taste graph, so every pick traces back to a favorite: a show, a game, an artist, or a book genre or trope tag. The model can only choose from what the taste graph returned and write about it; a grounding check drops any id that was not in the results. The "Why Qloo?" page shows both answers side by side.

## How the agent works

Qloo lookups are tool calls, run in a fixed plan, in parallel and within Qloo's rate limit; every step is shown in the trace:

| Tool | What it does |
|---|---|
| `resolve_entity(name)` | Find a favorite in the taste graph (a title, or a book genre / trope as Qloo tags) |
| `get_book_recs(entity_name)` | Books the graph links to that favorite (one Qloo call each, cached, retried on 429/5xx) |
| `get_blended_recs()` | Books that fit ALL the favorites together as one taste. A single genre or trope acts as a filter here: "Fourth Wing" + "dark romance" gives dark romance that Fourth Wing readers like |
| `finish(picks)` | The final three, each with a one-line reason |

Gemini then makes one call to choose the final three from what the taste graph returned and to say why, and a second call writes the message. Code filters out books the reader rejected and checks every pick against the taste-graph results. Two more rules shape the final three:

- **Spread:** if every pick came from one favorite (common, because books on both a favorite's list and the combined list rank first), later picks are swapped for well-known books from favorites no pick covers yet.
- **Well known:** a pick with under 10,000 Goodreads ratings is swapped for a better-known book from the same Qloo results. Someone in a slump does better with a well-loved book than a manga volume 22.

If Gemini is down or overloaded, the app falls back to a fixed rule-based choice and template text, and says so in the trace. Without a Gemini key the same fixed plan runs, so the app works offline. (`GeminiLLM(full_agent=True)` lets the model drive every tool call itself; it works but needs many slow round trips.)

The slump type (taste mismatch, burnout, too long, lost habit, nothing new) comes from what the business or reader says. When it is not given, the app assumes "lost the habit" and labels it a guess.

## Book genres and tropes

- **Book genres** use Qloo's own genre tags: books are filtered by the tag and ranked by popularity. Most genre names are found with Qloo's tag search ("cozy mystery", "psychological thriller"). Reader slang the search misses is mapped by hand (`GENRE_TAGS` in `slump_breaker/qloo.py`): *dark romance*, *romcom* / *rom com*, *romantasy*, *spicy*, *ya*, *sci fi*.
- **Tropes** use Qloo's book trope tags where they give good books (`TROPE_TAGS`): enemies to lovers, friends to lovers, second chance, fake dating, love triangle, arranged marriage, billionaire, small town, mafia romance, sports romance, workplace romance, reverse harem. Qloo's tag search does not list these for books, but books carry them.
- Tropes whose tags return only a few obscure books (grumpy sunshine, slow burn, found family, morally grey, forced proximity, chosen one…) start from two books famous for that trope (`_TROPES`). Books by those books' authors are left out, so a trope gives other writers, not sequels.
- Other spellings work: *rivals to lovers*, *Enemies-to-lovers*, *marriage of convenience*, *morally gray*, *only one bed* (`_TROPE_ALIASES`).
- Screen genres (*kdrama*, *anime*, *period drama*) are swapped for a couple of very well-known shows, because book tags do not describe them.

## Business CSV

```
name,favorites,reason
Maya,Stranger Things; Grand Theft Auto V; Taylor Swift,taste mismatch
Omar,Dune; Interstellar; Hans Zimmer,too long
Sana,Studio Ghibli; Minecraft; Harry Potter,
Ayesha,Fourth Wing; dark romance (genre); enemies to lovers (trope),burnout
```

`name` and `favorites` (separate with `;` or `|`) are required, `reason` is optional, up to 200 readers per run (set `MAX_READERS`, max 1000). File size is not the limit: the real limits are Qloo's request quota (10,000 a month, 5 a second on the hackathon key) and the free Gemini tier. Lists over 20 default to fast mode (template messages). Uploaded data is processed in the session only and never stored. Use made-up readers for demos. Exported CSV cells are neutralized against spreadsheet formula injection.

Test files in `tools/`:

- `full_system_test.csv`: 110 readers covering every kind of input, with a `notes` column saying what each should show.
- `test_readers.csv`: a short 33-row file for quick checks.

## Mock mode vs. live mode

Without a Qloo key the app runs on a small **illustrative fixture** (`slump_breaker/mock_data.py`). It is not real Qloo output and the UI says so. With `QLOO_API_KEY` set it uses the live API; if Qloo is unavailable it falls back to the fixture and labels the run. Without a Gemini key, messages use templates.

Before relying on live data, run `python tools/probe_qloo.py` (see the file header) to check the endpoints and response shapes in `slump_breaker/qloo.py` against the real API.

## Qloo facts this app relies on (verified against the hackathon API)

- Search results are mixed types ("Dune" returns a film, a book and a game). The app picks by closest name (ignoring series tags such as "(Twisted, #1)"), then prefers a far more popular match over an obscure namesake ("Twisted Love" is Ana Huang's book, not a 2026 film with no popularity), then by type (show, film, game, artist, book…). It expands short forms (`gta-5` becomes Grand Theft Auto V, `cod`, `hp`) and accepts a hint such as `Dune (book)`. The match is shown in the UI.
- Books come with page counts, covers, a Goodreads rating and genre tags. Box sets and omnibus editions are filtered out.
- `GET /v2/tags` finds book genre tags; insights with `filter.tags` returns the best-known books carrying them.
- Qloo's limits are 5 requests per second and 10,000 per month. The client spaces requests, caches for 6 hours, and retries 429/5xx with backoff.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add keys (optional)
streamlit run app.py
python -m pytest -q
```

Run `streamlit run app.py` from the project folder (the one with `app.py`). After changing code in `slump_breaker/`, stop the server (Ctrl+C) and start it again: a browser refresh only reloads `app.py`.

Keys can also be set as environment variables: `QLOO_API_KEY`, `GEMINI_API_KEY`, optionally `QLOO_BASE_URL`, `GEMINI_MODEL`. **Never commit keys.** `.streamlit/secrets.toml` is git-ignored.

## Sign-in (optional, free)

Create a free project at supabase.com, then in Project Settings > API copy the project URL and the `anon` public key into secrets as `SUPABASE_URL` and `SUPABASE_ANON_KEY`. In Authentication > Providers > Email you can turn off "Confirm email" for a smoother demo. Without these the app offers guest use only.

## Deploy (free)

Push to a public GitHub repo, create an app on Streamlit Community Cloud pointing at `app.py`, and paste the keys into the app's Secrets. Free apps sleep when idle; wake it before submitting and before judging.

## Input guide (for the best analysis)

- Use real, official titles: one per box, with its type. Choose **Book genre** or **Trope** for those.
- Mix kinds (a show, a game, an artist, a genre): the combined-taste match then has the most to work with.
- Three to five favorites per reader. More than five are cut off.
- Add the reason a reader stopped when you know it. A blank reason is labeled a guess.
- `python tools/check_gemini.py` checks the Gemini key and which models it can use.

## Scope and limits (stated honestly)

- It does not send messages, integrate with CRMs or predict churn, and the business supplies the list of lapsed readers. Win-back messages are suggestions; nothing is stored.
- Picks depend on Qloo's data. Some favorites have weak book lists (Taylor Swift's are business exposés; the Dune film's are One Piece volumes), so those readers lean on their other favorites. A missing book cannot be found: Qloo has no record of Ana Huang's *Twisted Lies*.
- Authors are found, but recommending from an author gives generic books; a specific title works better.
- A trope with no Qloo tag and no example books (e.g. *third act breakup*) is reported as not found rather than searched as a title.
- If Qloo does not know a title, the closest match is used and flagged for a second look (for example "sombr" -> "Sombra").
- Accounts are optional (Supabase email + password). Only role, business name, channel and tone are saved to your account; reader lists and favorites never are. A browser refresh signs you out.

## AI assistance

Built with help from Claude (Anthropic). Gemini is used for choosing picks and writing message text when a key is configured.

## License

MIT, see [LICENSE](LICENSE).
