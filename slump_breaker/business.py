"""Business view: turn a reader result into a win-back plan for a lapsed subscriber.

This is a proposed layer on top of taste data. It makes no claims about churn or
conversion; the evidence shown is exactly what the taste graph returned.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMError, MockLLM
from .models import AgentResult


@dataclass
class WinBackPlan:
    slump_label: str
    pick_title: str
    pick_author: str
    message: str
    evidence: list[str]
    message_source: str  # "gemini" or "mock"


def build_plan(result: AgentResult, reader_name: str, llm) -> WinBackPlan | None:
    if not result.picks:
        return None
    top = result.picks[0].candidate
    source = llm.mode
    try:
        message = llm.write_message(reader_name, top.name, top.author or "an author you may know", top.seeds, result.way_back)
    except LLMError:
        message = MockLLM().write_message(reader_name, top.name, top.author or "an author you may know", top.seeds, result.way_back)
        source = "mock"
    evidence = [f"Linked to the reader's favorite: {s}" for s in top.seeds]
    if top.tags:
        evidence.append("Shared themes: " + ", ".join(top.tags[:5]))
    if top.affinity is not None:
        evidence.append(f"Taste-graph affinity score: {top.affinity:.2f}")
    evidence.append(f"Data source: {'mock fixture (not real Qloo data)' if top.source == 'mock' else 'Qloo taste graph'}")
    return WinBackPlan(
        slump_label=result.slump_label,
        pick_title=top.name,
        pick_author=top.author,
        message=message,
        evidence=evidence,
        message_source=source,
    )
