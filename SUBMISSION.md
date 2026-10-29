# Slump Breaker: submission description

**Slump Breaker wins back lapsed readers with a comeback book chosen by Qloo's taste graph, not by guesswork.**

- **Live demo:** https://slump-breaker-xgmsapphsectx3mcq2nkr8z.streamlit.app/
- **Code:** https://github.com/EmanRohail/slump-breaker (MIT)
- **Try it:** Continue as guest → Reader view → type `enemies to lovers` (type: Trope), or upload `tools/test_readers.csv` in the Business view.

Book businesses (reading apps, book boxes, publishers, libraries) lose readers to reading slumps. A generic "here are this month's bestsellers" email rarely brings them back. Slump Breaker takes what a business already knows about a lapsed reader (a name, a few things they love, and optionally why they stopped) and, for each reader, finds a book that connects to their actual taste, then writes a personal win-back message for email, push or SMS.

## What it does

- **Business view:** upload a CSV of lapsed readers (up to 200 per run) or add one by hand. For each reader the agent returns a comeback book, two alternatives, the evidence linking them to the reader's favorites, and a ready-to-send message in the chosen channel and tone. Results download as a CSV.
- **Reader view:** the same engine from the reader's side. Enter up to five favorites and why you stopped reading, and get three picks with covers, ratings and a reason. "Worked / Not for me" feedback re-asks Qloo and refills the picks.
- **Why Qloo? view:** the same reader asked two ways, a plain AI next to the Qloo-grounded answer, so the difference is visible.

Favorites can be anything a reader loves: **TV shows, movies, video games, music artists, books, book genres** (dark romance, romcom, romantasy, cozy mystery) **and tropes** (enemies to lovers, fake dating, second chance, found family). Auto-detect works out the type, and a dropdown lets the reader say it exactly.

## What makes it Qloo-powered

Every book comes from Qloo; the AI never invents one.

1. **Cross-domain taste.** Qloo links a reader's favorite show, game or artist to books that fans of it read. "Stranger Things" leads to *The First to Die at the End*; "Hans Zimmer" leads to *The Book Thief*. A plain LLM has no such link and suggests popular books.
2. **Combined taste.** Qloo is asked about each favorite alone and about all favorites together as one taste signal. Books that fit the whole mix rank first.
3. **Genres and tropes as Qloo tags.** "dark romance", "romantasy" or "enemies to lovers" become Qloo book tags, and the books are the best-known ones carrying them. Combined with a title, a genre filters that title's taste: "Fourth Wing" + "dark romance" gives dark romance that Fourth Wing readers love.
4. **Grounded output.** Gemini only chooses among the books Qloo returned and explains why. A grounding check drops anything not in Qloo's results, and every pick shows which favorite it came from.

## The agent

A bounded tool-calling loop: `resolve_entity` → `get_book_recs` per favorite → `get_blended_recs` → `finish`. Lookups run in parallel within Qloo's rate limit, are cached and retried, and every step is shown in a trace in the app. The code then:

- spreads picks across the reader's favorites (so one favorite does not take all three),
- prefers well-known books over obscure ones (a slump reader needs a book likely to land),
- leaves out books by a trope's example authors (so a trope gives other writers, not sequels),
- and honours "Not for me" feedback.

Title matching is tuned to Qloo's real data. It ignores series tags such as "(Twisted, #1)", prefers a famous book over an obscure namesake ("Twisted Love" is Ana Huang's novel, not a 2026 film), and expands short forms like "gta-5".

If Gemini is unavailable, a fixed rule-based plan runs the same Qloo tools. If Qloo is unavailable, clearly labelled sample data is used. The app never fails silently.

## Built with

Python, Streamlit, the Qloo Insights API (search, tags, insights), Google Gemini (function calling and message text), optional Supabase sign-in. MIT licensed.
