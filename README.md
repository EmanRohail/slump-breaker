# Slump Breaker

A reading-slump recovery agent grounded in [Qloo](https://qloo.com)'s taste graph. Built for the Qloo Agentic Hackathon.

A reader lists a few things they love (shows, games, music, films, books) and says why they stopped reading. Slump Breaker diagnoses the slump type, asks the taste graph for books linked to those favorites, and picks three with a plan for getting back in. A second tab turns the same result into a **win-back plan** for a reading app, book box or publisher: a ready-to-send re-engagement message plus the evidence behind the pick.

**Live demo:** _add your Streamlit Community Cloud URL here_

## Why it needs Qloo

A plain LLM recommends popular books. It has no link between *this* reader's favorites and the books it suggests. Here the candidate list comes from Qloo's taste graph, so every pick traces back to a favorite the reader named. The LLM can only choose from that list and write the explanation; it cannot add books. A grounding check drops any id that is not in the Qloo results (see `tests/test_agent.py`). The "With vs. without Qloo" tab shows both side by side.

## How it works

1. **Diagnose** the slump type (taste mismatch, burnout, books too long, lost habit, nothing new).
2. **Resolve** each favorite to a taste-graph entity (`/search`).
3. **Recommend** books per favorite (`/v2/insights`, book entity type), keeping which favorite linked to which book.
4. **Select** three with the LLM, from that list only.
5. **Ground-check** the selection, then attach the slump type's way back in.
6. **Feedback loop:** "Not for me" excludes a book and re-runs.

Every step is recorded in a trace, shown in the app under "How the agent got here". Note this is a fixed pipeline with a trace, not a model-driven tool-calling loop.

## Mock mode vs. live mode

Without a Qloo key the app runs on a small **illustrative fixture** (`slump_breaker/mock_data.py`). It is not real Qloo output and the UI says so. With `QLOO_API_KEY` set it uses the live API; if Qloo is unavailable (rate limit, outage) it falls back to the fixture and labels the run `live-fallback-mock`. Without a Gemini key, explanations use a deterministic template.

The Qloo endpoint paths and response shapes in `slump_breaker/qloo.py` were written from public examples and should be checked against the live API once a key is available.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add keys (optional)
streamlit run app.py
python -m pytest -q
```

Keys can also be set as environment variables: `QLOO_API_KEY`, `GEMINI_API_KEY`, optionally `QLOO_BASE_URL`, `GEMINI_MODEL`.

## Deploy (free)

Push to a public GitHub repo, create an app on Streamlit Community Cloud pointing at `app.py`, and paste the keys into the app's Secrets. Free apps go to sleep when idle; wake it before submitting and again before judging.

## Scope and limits

- The business tab is a proposed layer on top of taste data. It makes no claims about churn or conversion.
- Book metadata (pages, author) depends on what the taste graph returns.
- Feedback is kept per session only.

## AI assistance

This project was built with help from Claude (Anthropic) and uses Gemini for text generation when a key is configured.

## License

MIT, see [LICENSE](LICENSE).
