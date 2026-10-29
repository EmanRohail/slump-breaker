"""Plain data containers shared across the app."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """A favorite resolved to a taste-graph entity."""

    id: str
    name: str
    type: str = ""
    source: str = "qloo"  # "qloo" or "mock"
    author: str = ""  # books only, from Qloo's disambiguation


@dataclass
class Candidate:
    """A book the taste graph suggests, with the evidence behind it."""

    id: str
    name: str
    author: str = ""
    tags: list[str] = field(default_factory=list)
    affinity: Optional[float] = None
    seeds: list[str] = field(default_factory=list)  # favorites whose own list contained this book
    blend: list[str] = field(default_factory=list)  # favorites in the combined-taste query that returned it
    pages: Optional[int] = None  # only known for some sources
    source: str = "qloo"
    image_url: str = ""  # cover from Qloo, empty when none
    year: Optional[int] = None
    rating: Optional[float] = None  # Goodreads rating as reported by Qloo
    rating_count: Optional[int] = None

    @property
    def fits(self) -> list[str]:
        """Every favorite this book connects to: its own list plus the combined-taste query."""
        return list(dict.fromkeys([*self.seeds, *self.blend]))


@dataclass
class Pick:
    """A final recommendation shown to the reader."""

    candidate: Candidate
    why: str


@dataclass
class ToolCall:
    """One tool the agent asked to run (resolve_entity, get_book_recs, finish)."""

    name: str
    args: dict = field(default_factory=dict)


@dataclass
class TraceStep:
    tool: str
    detail: str
    ok: bool = True


@dataclass
class AgentResult:
    slump_key: str
    slump_label: str
    way_back: str
    picks: list[Pick]
    unresolved: list[str]
    trace: list[TraceStep]
    mode: str  # "live", "mock", or "live-fallback-mock"
    llm_mode: str  # "gemini" or "mock"
    matches: list[str] = field(default_factory=list)  # "gta-5 -> Grand Theft Auto V (videogame)"
    resolved: list[str] = field(default_factory=list)  # names of the favorites that were found
    genre_of: dict[str, str] = field(default_factory=dict)  # stand-in title -> the genre word the reader typed
    uncertain: list[str] = field(default_factory=list)  # 'typed -> matched' pairs worth a second look


def fallback_reason(result: "AgentResult") -> str:
    """Why Qloo data was replaced by sample data, in the words the trace recorded. Empty if it was not."""
    for step in result.trace:
        if step.tool == "taste_graph" and not step.ok:
            return step.detail
    return ""


def blend_problem(result: "AgentResult") -> str:
    """Why the combined-taste step was skipped, or an empty string if it ran (or was not needed)."""
    for step in result.trace:
        if step.tool == "get_blended_recs" and not step.ok:
            return step.detail
    return ""


def spoken(names, genre_of: dict) -> list:
    """Use the reader's own words: stand-in titles for 'kdrama' are reported as 'kdrama', once."""
    return list(dict.fromkeys(genre_of.get(n, n) for n in names))
