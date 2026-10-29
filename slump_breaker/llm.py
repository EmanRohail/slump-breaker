"""LLM layer: Gemini over REST when a key exists, a deterministic mock otherwise.

The agent runs as a short tool-calling loop (see agent.py). A "session" decides
which tool to call next. Default with Gemini (HybridSession): code runs the lookups in a fixed plan and Gemini
makes one call to choose the final picks. GeminiSession (full_agent=True) lets the model drive every call through
function calling. MockSession follows the fixed plan with a rule-based choice, so the app works offline.

The LLM never invents books. It only chooses among candidates that came from the
taste graph, and the agent drops any id that is not in that list.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Callable, Optional

import requests

from . import mock_data
from .models import Candidate, ToolCall
from .slump_types import SlumpType


class LLMError(Exception):
    pass


CHANNEL_LIMITS = {
    "email": "at most 70 words",
    "push": "at most 20 words, one line",
    "sms": "at most 35 words, plain text",
}
TONES = {
    "warm": "warm and kind",
    "playful": "light and playful",
    "brief": "short and direct",
}

# Tool schemas shown to the model. The executor lives in agent.py.
TOOL_DECLARATIONS = [
    {
        "name": "resolve_entity",
        "description": "Look up one of the reader's favorites in the taste graph. Call once per favorite.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The favorite exactly as the reader typed it."}},
            "required": ["name"],
        },
    },
    {
        "name": "get_book_recs",
        "description": "Get books the taste graph links to one resolved favorite. Use the name returned by resolve_entity.",
        "parameters": {
            "type": "object",
            "properties": {"entity_name": {"type": "string"}},
            "required": ["entity_name"],
        },
    },
    {
        "name": "get_blended_recs",
        "description": "Get books that fit ALL resolved favorites together as one taste. Call once, after get_book_recs, when two or more favorites resolved.",
    },
    {
        "name": "finish",
        "description": "Give the final 3 picks. Ids must come from get_book_recs or get_blended_recs results.",
        "parameters": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "why": {"type": "string", "description": "One sentence using only linked_favorites and tags."},
                        },
                        "required": ["id", "why"],
                    },
                }
            },
            "required": ["picks"],
        },
    },
]


def _short(c: Candidate) -> bool:
    return c.pages is not None and c.pages <= 300


def _template_why(c: Candidate) -> str:
    if c.blend and not c.seeds:
        themes = ", ".join(c.tags[:3])
        base = f"A strong fit for your whole mix: {', '.join(c.blend)}."
        return f"{base} Shared themes: {themes}." if themes else base
    seeds = ", ".join(c.seeds) if c.seeds else "your favorites"
    themes = ", ".join(c.tags[:3])
    base = f"Taste-graph readers who love {seeds} also love it."
    return f"{base} Shared themes: {themes}." if themes else base


class MockSession:
    """Fixed plan: resolve every favorite, fetch books for each, then finish."""

    def __init__(self, llm: "MockLLM", task: dict) -> None:
        self.llm = llm
        self.task = task
        self.phase = 0

    def next_calls(self, results: Optional[list], pool: list[Candidate]) -> list[ToolCall]:
        if self.phase == 0:
            self.phase = 1
            return [ToolCall("resolve_entity", {"name": f}) for f in self.task["favorites"]]
        if self.phase == 1:
            self.phase = 2
            names = [r["entity"]["name"] for _, r in (results or []) if r.get("found")]
            if not names:
                return [ToolCall("finish", {"picks": []})]
            calls = [ToolCall("get_book_recs", {"entity_name": n}) for n in names]
            if len(names) >= 2:
                calls.append(ToolCall("get_blended_recs", {}))
            return calls
        picks = self.llm.select_picks(pool, self.task["slump"], self.task["favorites"], self.task["minutes"])
        return [ToolCall("finish", {"picks": picks})]


class MockLLM:
    mode = "mock"

    def new_session(self, task: dict) -> MockSession:
        return MockSession(self, task)

    def select_picks(
        self,
        candidates: list[Candidate],
        slump: SlumpType,
        favorites: list[str],
        minutes: Optional[int],
    ) -> list[dict]:
        want_short = slump.prefers_short or (minutes is not None and minutes <= 15)

        def score(c: Candidate) -> tuple:
            short_bonus = 1 if (want_short and _short(c)) else 0
            return (-short_bonus, -len(c.fits), -(c.affinity or 0.0), c.name)

        ranked = sorted(candidates, key=score)
        return [{"id": c.id, "why": _template_why(c)} for c in ranked[:3]]

    def plain_recs(self, favorites: list[str], reason_label: str) -> list[dict]:
        return [{"title": t, "author": a} for t, a in mock_data.MOCK_GENERIC_PICKS]

    def write_message(
        self,
        name: str,
        title: str,
        author: str,
        seeds: list[str],
        way_back: str,
        channel: str = "email",
        tone: str = "warm",
        business: str = "",
    ) -> str:
        loves = ", ".join(seeds) if seeds else "what you love"
        sign = f" - {business}" if business and channel == "email" else ""
        if channel == "push":
            return f"{name}, your next read: {title}. Just 10 minutes."
        if channel == "sms":
            prefix = f"{business}: " if business else ""
            return f"{prefix}{name}, since you love {loves}, try {title} by {author}. 10 minutes is plenty."
        if tone == "brief":
            return f"Hi {name}, since you love {loves}, try {title} by {author}. Ten minutes is plenty.{sign}"
        if tone == "playful":
            return (
                f"Hi {name}! Plot twist: the thing you love in {loves} has a book. "
                f"Try {title} by {author}. {way_back} Ten minutes, no pressure.{sign}"
            )
        return (
            f"Hi {name}, it has been a while, so here is one small thing. Since you love {loves}, "
            f"you might like {title} by {author}. {way_back} No pressure, ten minutes is plenty.{sign}"
        )


def _agent_prompt(task: dict) -> str:
    slump: SlumpType = task["slump"]
    minutes = task.get("minutes")
    return (
        "You are an agent that finds books for a reader who has stopped reading, using a taste graph.\n"
        "Always respond by calling a tool. Never answer in plain text.\n"
        "Plan: call resolve_entity for EVERY favorite (several calls at once is fine), then call get_book_recs "
        "for each favorite that resolved, then (if two or more resolved) get_blended_recs once, then call finish with exactly 3 picks.\n"
        "Choose picks ONLY from ids returned by get_book_recs or get_blended_recs. Prefer books that fit several favorites, "
        "especially those in get_blended_recs.\n"
        "For each pick, write one sentence 'why' that uses ONLY linked_favorites and tags from the tool results. "
        "Do not add plot facts, ratings or any claim that is not in the data.\n\n"
        f"Reader's favorites: {task['favorites']}\n"
        f"Why they stopped reading: {slump.label}\n"
        f"Time per day: {minutes if minutes else 'unspecified'} minutes\n"
        + ("This reader needs shorter, lighter reads: favor them when the data shows length.\n" if slump.prefers_short else "")
    )


class GeminiSession:
    """The model chooses the tools; we feed each tool's result back to it."""

    def __init__(self, llm: "GeminiLLM", task: dict) -> None:
        self.llm = llm
        self.contents: list[dict] = [{"role": "user", "parts": [{"text": _agent_prompt(task)}]}]

    def next_calls(self, results: Optional[list], pool: list[Candidate]) -> list[ToolCall]:
        if results:
            parts = [{"functionResponse": {"name": call.name, "response": res}} for call, res in results]
            self.contents.append({"role": "user", "parts": parts})
        model_content = self.llm._function_turn(self.contents)
        self.contents.append(model_content)
        calls = []
        for part in model_content.get("parts", []):
            fc = part.get("functionCall")
            if fc and fc.get("name"):
                calls.append(ToolCall(str(fc["name"]), dict(fc.get("args") or {})))
        if not calls:
            raise LLMError("model returned no tool call")
        return calls


class HybridSession(MockSession):
    """Code runs the lookups (fast, in parallel, same tools and trace); Gemini then makes ONE call to choose the
    final three from what the taste graph returned. A model that drives every lookup itself needs a dozen slow
    round trips; this needs one."""

    def next_calls(self, results: Optional[list], pool: list[Candidate]) -> list[ToolCall]:
        if self.phase < 2:
            return MockSession.next_calls(self, results, pool)
        return [ToolCall("finish", {"picks": self.llm.choose_picks(pool, self.task)})]


class GeminiLLM:
    mode = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-lite-latest",
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        full_agent: bool = False,
    ) -> None:
        self.full_agent = full_agent  # True: the model decides every tool call itself (slow, many round trips)
        self._no_thinking = False
        self.api_key = api_key
        self.model = model
        self.session = session or requests.Session()
        self._sleep = sleep
        self._failed_at = 0.0
        self._tl = threading.local()  # each request (and each parallel reader) has its own time budget
        self._last_error = ""
        self._cache: dict[str, object] = {}  # same input -> same answer, no new model call (saves the free quota)
        self._cache_lock = threading.Lock()

    def _cached(self, key: str, make: Callable[[], object]):
        k = hashlib.sha256(key.encode()).hexdigest()
        with self._cache_lock:
            if k in self._cache:
                return self._cache[k]
        value = make()
        with self._cache_lock:
            if len(self._cache) >= 500:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = value
        return value

    # ------------------------------------------------------------------ public

    def new_session(self, task: dict):
        return GeminiSession(self, task) if self.full_agent else HybridSession(self, task)

    def choose_picks(self, pool: list[Candidate], task: dict) -> list[dict]:
        """One model call: the three best books from the taste-graph results, each with a one-line reason."""
        top = pool[:12]
        if not top:
            return []
        slump: SlumpType = task["slump"]
        books = [{
            "id": c.id, "title": c.name, "author": c.author, "tags": c.tags[:6], "affinity": c.affinity,
            "linked_favorites": c.seeds, "fits_combined_taste_of": c.blend, "pages": c.pages,
        } for c in top]
        prompt = (
            "A reader stopped reading. From the books below (all returned by a taste graph), choose exactly 3 and "
            "call finish. Use ONLY the ids listed. Prefer books that fit several of the reader's favorites. "
            "For each, write one sentence 'why' using ONLY linked_favorites, fits_combined_taste_of and tags: "
            "no plot facts, ratings or other claims.\n"
            f"Reader's favorites: {task['favorites']}\nWhy they stopped: {slump.label}\n"
            f"Time per day: {task.get('minutes') or 'unspecified'} minutes\n"
            + ("Favor shorter, lighter books when the data shows length.\n" if slump.prefers_short else "")
            + "Books: " + json.dumps(books)
        )
        finish = next(d for d in TOOL_DECLARATIONS if d["name"] == "finish")
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"functionDeclarations": [finish]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["finish"]}},
        }
        return self._cached("picks|" + prompt, lambda: self._choose(body))

    def _choose(self, body: dict) -> list[dict]:
        self._fast(body)
        data = self._post(body)
        try:
            for part in data["candidates"][0]["content"]["parts"]:
                fc = part.get("functionCall")
                if fc and fc.get("name") == "finish":
                    picks = (fc.get("args") or {}).get("picks")
                    if isinstance(picks, list):
                        return [p for p in picks if isinstance(p, dict)]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Gemini response shape") from exc
        raise LLMError("model did not choose picks")

    def _fast(self, body: dict) -> None:
        """Ask for little or no hidden 'thinking': these tasks are small and thinking only adds seconds."""
        if not self._no_thinking:
            body.setdefault("generationConfig", {})["thinkingConfig"] = {"thinkingBudget": 0}

    def plain_recs(self, favorites: list[str], reason_label: str) -> list[dict]:
        prompt = (
            f"A reader loves {favorites} and is in a reading slump ({reason_label}). "
            "Recommend 3 books. Return JSON only: "
            '[{"title": "...", "author": "..."}]'
        )
        data = self._json(prompt)
        if not isinstance(data, list):
            raise LLMError("expected a JSON list")
        return [
            {"title": str(d.get("title", "")), "author": str(d.get("author", ""))}
            for d in data[:3]
            if isinstance(d, dict)
        ]

    def write_message(
        self,
        name: str,
        title: str,
        author: str,
        seeds: list[str],
        way_back: str,
        channel: str = "email",
        tone: str = "warm",
        business: str = "",
    ) -> str:
        prompt = (
            f"Write a {TONES.get(tone, TONES['warm'])} {channel} message ({CHANNEL_LIMITS.get(channel, CHANNEL_LIMITS['email'])}) "
            "from a reading business to a lapsed reader.\n"
            f"Business name: {business or 'not given'}\n"
            f"Reader name: {name}\nTheir favorites: {seeds}\nSuggested book: {title} by {author}\n"
            f"Way back in: {way_back}\n"
            "Use only these facts. No discounts, no stats, no claims about the book beyond the title and author. "
            "Return plain text only."
        )
        return self._cached("msg|" + prompt, lambda: self._text(prompt).strip())

    # ----------------------------------------------------------------- private

    # Model names are retired from time to time. If one is "not found" (404), try these in turn and keep the one that works.
    _FALLBACK_MODELS = ("gemini-flash-latest",)  # an alias that stays valid; real names come from the key itself

    BUDGET_SECONDS = 25  # the most one model request may take in total, across retries and model names

    def _left(self) -> float:
        return getattr(self._tl, "deadline", time.time() + self.BUDGET_SECONDS) - time.time()

    def _try_models(self, models: list[str], body: dict, headers: dict) -> tuple[Optional[dict], str, str]:
        """Try model names in turn. Returns (answer or None, last transient problem, not-found note)."""
        last = "unknown"
        not_found = ""
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            for attempt in range(2):  # one retry: a busy service rarely recovers in seconds, so fall back fast
                if self._left() <= 1:
                    return None, f"{last if last != 'unknown' else 'no answer'} (time budget of {self.BUDGET_SECONDS}s used)", not_found
                try:
                    resp = self._send(url, headers, body)
                except requests.RequestException as exc:
                    last = f"network error: {exc.__class__.__name__}"
                    break  # a hung or unreachable model will not recover in a second: next model
                else:
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except ValueError as exc:
                            raise LLMError("Gemini returned invalid JSON") from exc
                        self.model = model  # remember the one that works
                        return data, last, not_found
                    if resp.status_code == 404:
                        not_found = f"model {model} not found"
                        break  # try the next model name
                    if resp.status_code == 429:  # this model's free quota is used up; every model has its own, so try the next
                        last = "quota"
                        break
                    if resp.status_code in (500, 502, 503):
                        last = f"Google is busy (HTTP {resp.status_code})"
                        break  # a busy model rarely recovers in a second: try the next model name
                    elif resp.status_code == 400:  # this model dislikes the request: another model may accept it
                        text = str(getattr(resp, "text", "") or "").replace("\n", " ")[:160]
                        last = f"HTTP 400 on {model}" + (f": {text}" if text else "")
                        break
                    else:
                        text = str(getattr(resp, "text", "") or "").replace("\n", " ")[:160]
                        raise LLMError(f"HTTP {resp.status_code} from Gemini" + (f": {text}" if text else ""))
                if attempt < 1:
                    self._sleep(1.5)
            # retries on this model are used up (overloaded or over its quota): each model has its own capacity, so try the next
        return None, last, not_found

    def _send(self, url: str, headers: dict, body: dict):
        timeout = max(2.0, min(12.0, self._left()))
        resp = self.session.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code == 400:
            cfg = body.get("generationConfig") or {}
            if "thinkingConfig" in cfg:  # a model may reject this setting (often with a vague 'invalid argument'): drop it, remember, retry
                del cfg["thinkingConfig"]
                if not cfg:
                    body.pop("generationConfig", None)
                self._no_thinking = True
                resp = self.session.post(url, headers=headers, json=body, timeout=timeout)
        return resp

    def _discover_models(self, headers: dict) -> list[str]:
        """Ask Gemini which models this key can use for generateContent (flash ones first, newest first)."""
        if self._left() < 3:
            return []
        try:
            resp = self.session.get("https://generativelanguage.googleapis.com/v1beta/models",
                                    params={"pageSize": 100}, headers=headers, timeout=min(5.0, self._left()))
            if resp.status_code != 200:
                return []
            items = resp.json().get("models", [])
        except (requests.RequestException, ValueError, AttributeError, TypeError):
            return []
        skip = ("image", "tts", "live", "audio", "embedding", "thinking", "robotics", "computer", "vision")
        names = []
        for m in items:
            if not isinstance(m, dict) or "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            n = str(m.get("name", "")).removeprefix("models/")
            if "flash" in n and not any(s in n for s in skip):
                names.append(n)
        # stable models before preview/experimental ones, then newest name first
        names.sort(reverse=True)
        names.sort(key=lambda n: (any(x in n for x in ("preview", "exp")), "lite" in n))
        return names[:6]

    def _post(self, body: dict) -> dict:
        # After a failure, skip Gemini for a short while so one slow or broken service cannot stall every step.
        if time.time() - self._failed_at < 120:
            raise LLMError(f"Gemini paused for 2 minutes after: {self._last_error}")
        self._tl.deadline = time.time() + self.BUDGET_SECONDS
        try:
            return self._post_inner(body)
        except LLMError as exc:
            self._failed_at, self._last_error = time.time(), str(exc)
            raise

    def _post_inner(self, body: dict) -> dict:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        models = list(dict.fromkeys([self.model, *self._FALLBACK_MODELS]))
        data, last, not_found = self._try_models(models, body, headers)
        if data is not None:
            return data
        # The names we know are gone or busy: ask Gemini which models THIS key can use, and try those.
        extra = [m for m in self._discover_models(headers) if m not in models]
        if extra:
            data, last2, nf2 = self._try_models(extra, body, headers)
            if data is not None:
                return data
            models += extra
            last, not_found = (last2 if last2 != "unknown" else last), (nf2 or not_found)
        if last == "quota":
            raise LLMError("Gemini's free quota is used up for now (HTTP 429); try again in a few minutes")
        if not_found and last == "unknown":
            raise LLMError(f"HTTP 404 from Gemini ({not_found}; tried {', '.join(models)})")
        raise LLMError(f"Gemini unavailable ({last}; tried {', '.join(models)})")

    def _function_turn(self, contents: list[dict]) -> dict:
        body = {
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        }
        data = self._post(body)
        try:
            content = data["candidates"][0]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Gemini response shape") from exc
        if not isinstance(content, dict) or not content.get("parts"):
            raise LLMError("empty Gemini turn")
        content.setdefault("role", "model")
        return content

    def _text(self, prompt: str, json_mode: bool = False) -> str:
        body: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        self._fast(body)
        data = self._post(body)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Gemini response shape") from exc

    def _json(self, prompt: str) -> object:
        raw = self._text(prompt, json_mode=True)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini did not return valid JSON") from exc
