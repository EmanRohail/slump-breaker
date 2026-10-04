"""LLM layer: Gemini over REST when a key exists, a deterministic mock otherwise.

The LLM never invents books. It only chooses among candidates that came from
the taste graph, and the agent drops any id that is not in that list.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

import requests

from . import mock_data
from .models import Candidate
from .slump_types import SlumpType


class LLMError(Exception):
    pass


def _short(c: Candidate) -> bool:
    return c.pages is not None and c.pages <= 300


def _template_why(c: Candidate) -> str:
    seeds = ", ".join(c.seeds) if c.seeds else "your favorites"
    themes = ", ".join(c.tags[:3])
    base = f"Taste-graph readers who love {seeds} also love it."
    return f"{base} Shared themes: {themes}." if themes else base


class MockLLM:
    mode = "mock"

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
            return (-short_bonus, -len(c.seeds), -(c.affinity or 0.0), c.name)

        ranked = sorted(candidates, key=score)
        return [{"id": c.id, "why": _template_why(c)} for c in ranked[:3]]

    def plain_recs(self, favorites: list[str], reason_label: str) -> list[dict]:
        return [{"title": t, "author": a} for t, a in mock_data.MOCK_GENERIC_PICKS]

    def write_message(self, name: str, title: str, author: str, seeds: list[str], way_back: str) -> str:
        loves = ", ".join(seeds) if seeds else "what you love"
        return (
            f"Hi {name}, it has been a while, so here is one small thing. Since you love {loves}, "
            f"you might like {title} by {author}. {way_back} No pressure, ten minutes is plenty."
        )


class GeminiLLM:
    mode = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.session = session or requests.Session()
        self._sleep = sleep

    # ------------------------------------------------------------------ public

    def select_picks(
        self,
        candidates: list[Candidate],
        slump: SlumpType,
        favorites: list[str],
        minutes: Optional[int],
    ) -> list[dict]:
        listing = [
            {"id": c.id, "title": c.name, "author": c.author, "tags": c.tags[:6], "linked_favorites": c.seeds}
            for c in candidates[:20]
        ]
        prompt = (
            "You help a reader get out of a reading slump.\n"
            f"Reader's favorites: {favorites}\n"
            f"Slump type: {slump.label}\n"
            f"Time per day: {minutes if minutes else 'unspecified'} minutes\n"
            "Candidate books from a taste graph (the ONLY books you may choose from):\n"
            f"{json.dumps(listing)}\n\n"
            "Choose exactly 3 candidates that best help this reader back into reading"
            + (", favoring shorter, lighter reads. " if slump.prefers_short else ". ")
            + "For each, write one sentence 'why' that uses ONLY the linked_favorites and tags given. "
            "Do not add plot facts, ratings or claims that are not in the data.\n"
            'Return JSON only: [{"id": "<candidate id>", "why": "<one sentence>"}]'
        )
        data = self._json(prompt)
        if not isinstance(data, list):
            raise LLMError("expected a JSON list of picks")
        return [{"id": str(d.get("id", "")), "why": str(d.get("why", ""))} for d in data if isinstance(d, dict)]

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

    def write_message(self, name: str, title: str, author: str, seeds: list[str], way_back: str) -> str:
        prompt = (
            "Write a warm, short (max 70 words) re-engagement message from a reading app to a lapsed reader.\n"
            f"Reader name: {name}\nTheir favorites: {seeds}\nSuggested book: {title} by {author}\n"
            f"Way back in: {way_back}\n"
            "Use only these facts. No discounts, no stats, no claims about the book beyond the title and author. "
            "Return plain text only."
        )
        return self._text(prompt).strip()

    # ----------------------------------------------------------------- private

    def _text(self, prompt: str, json_mode: bool = False) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        last = "unknown"
        for attempt in range(3):
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=40)
            except requests.RequestException as exc:
                last = f"network error: {exc.__class__.__name__}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise LLMError("unexpected Gemini response shape") from exc
                if resp.status_code in (429, 500, 502, 503):
                    last = f"HTTP {resp.status_code}"
                else:
                    raise LLMError(f"HTTP {resp.status_code} from Gemini")
            if attempt < 2:
                self._sleep(1.5 * (2**attempt))
        raise LLMError(f"Gemini unavailable ({last})")

    def _json(self, prompt: str) -> object:
        raw = self._text(prompt, json_mode=True)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini did not return valid JSON") from exc
