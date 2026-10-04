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


@dataclass
class Candidate:
    """A book the taste graph suggests, with the evidence behind it."""

    id: str
    name: str
    author: str = ""
    tags: list[str] = field(default_factory=list)
    affinity: Optional[float] = None
    seeds: list[str] = field(default_factory=list)  # favorites that led here
    pages: Optional[int] = None  # only known for some sources
    source: str = "qloo"


@dataclass
class Pick:
    """A final recommendation shown to the reader."""

    candidate: Candidate
    why: str


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
