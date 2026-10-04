"""The one place the app talks to Qloo.

Everything else calls QlooClient.resolve() and QlooClient.book_recs(), so when
the real API key arrives only this file needs checking.

IMPORTANT, UNVERIFIED: the endpoints, parameter names and response shapes below
come from other entrants' public READMEs (base URL hackathon.api.qloo.com, an
X-Api-Key header, GET /search and GET /v2/insights with filter.type and
signal.interests.entities). They have NOT been tested against a real key. The
_parse_* helpers are written defensively; adjust them after the first live call.

With no API key the client runs in mock mode using fixtures from mock_data.py.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import requests

from . import mock_data
from .models import Candidate, Entity

CACHE_TTL_SECONDS = 6 * 60 * 60


class QlooUnavailable(Exception):
    """Raised when Qloo cannot be reached after retries (rate limit, outage)."""


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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._clock = clock
        self._cache: dict[tuple, tuple[float, object]] = {}
        self.calls_made = 0  # real HTTP requests, for tests and the trace

    @property
    def is_mock(self) -> bool:
        return not self.api_key

    # ------------------------------------------------------------------ public

    def resolve(self, name: str) -> Optional[Entity]:
        """Resolve a free-text favorite to an entity, or None if unrecognized."""
        name = name.strip()
        if not name:
            return None
        if self.is_mock:
            hit = mock_data.MOCK_ENTITIES.get(name.lower())
            if not hit:
                return None
            display, etype = hit
            return Entity(id=f"mock:{display.lower()}", name=display, type=etype, source="mock")
        data = self._get("/search", {"query": name, "take": 1})
        entities = _extract_entities(data)
        if not entities:
            return None
        e = entities[0]
        eid = e.get("entity_id") or e.get("id")
        if not eid:
            return None
        return Entity(id=str(eid), name=e.get("name") or name, type=_entity_type(e), source="qloo")

    def book_recs(self, seeds: list[Entity], per_seed: int = 8) -> list[Candidate]:
        """Books the taste graph links to the seeds, merged and ranked.

        One call per seed, so every book keeps the list of favorites that led to
        it. That linkage is the evidence shown to the user.
        """
        merged: dict[str, Candidate] = {}
        for seed in seeds:
            for cand in self._recs_for_seed(seed, per_seed):
                existing = merged.get(cand.id)
                if existing is None:
                    merged[cand.id] = cand
                else:
                    existing.seeds = sorted(set(existing.seeds) | set(cand.seeds))
                    existing.tags = sorted(set(existing.tags) | set(cand.tags))
                    if cand.affinity is not None:
                        prev = existing.affinity if existing.affinity is not None else cand.affinity
                        existing.affinity = max(prev, cand.affinity)
        ranked = list(merged.values())
        # Books linked to more of the reader's favorites come first, then by affinity.
        ranked.sort(key=lambda c: (-len(c.seeds), -(c.affinity or 0.0), c.name))
        return ranked

    # ----------------------------------------------------------------- private

    def _recs_for_seed(self, seed: Entity, take: int) -> list[Candidate]:
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
        params = {
            "filter.type": "urn:entity:book",
            "signal.interests.entities": seed.id,
            "take": take,
        }
        data = self._get("/v2/insights", params)
        out: list[Candidate] = []
        for e in _extract_entities(data):
            eid = e.get("entity_id") or e.get("id")
            if not eid:
                continue
            out.append(
                Candidate(
                    id=str(eid),
                    name=e.get("name", "Unknown title"),
                    author=_author(e),
                    tags=_tag_names(e),
                    affinity=_affinity(e),
                    seeds=[seed.name],
                    pages=None,
                    source="qloo",
                )
            )
        return out

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
                    raise QlooUnavailable(f"HTTP {resp.status_code} from {path}")
            if attempt < self.max_retries:
                self._sleep(self.backoff_seconds * (2**attempt))
        raise QlooUnavailable(f"{path} failed after retries ({last_error})")


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
    types = e.get("types") or e.get("type") or []
    if isinstance(types, list):
        return str(types[0]) if types else ""
    return str(types)


def _tag_names(e: dict) -> list[str]:
    tags = e.get("tags") or []
    names = []
    for t in tags:
        if isinstance(t, dict) and t.get("name"):
            names.append(str(t["name"]))
        elif isinstance(t, str):
            names.append(t)
    return names


def _author(e: dict) -> str:
    props = e.get("properties") or {}
    value = props.get("author") or props.get("authors") or ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _affinity(e: dict) -> Optional[float]:
    q = e.get("query") or {}
    value = q.get("affinity") if isinstance(q, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
