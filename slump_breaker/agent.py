"""The Slump Breaker agent flow.

1. Resolve each favorite to a taste-graph entity.
2. Ask the taste graph for books linked to those entities.
3. Let the LLM choose 3 from THAT list only, and explain with the linked data.
4. Check the choice is grounded, then attach the slump type's way back in.

Every step is recorded in a trace the UI can show, so judges can see the agent work.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .llm import GeminiLLM, LLMError, MockLLM, _template_why
from .models import AgentResult, Candidate, Entity, Pick, TraceStep
from .qloo import QlooClient, QlooUnavailable
from .slump_types import get_slump

MAX_FAVORITES = 5


def clean_favorites(raw: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        name = (item or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out[:MAX_FAVORITES]


def _resolve_all(client: QlooClient, favorites: list[str], trace: list[TraceStep]) -> tuple[list[Entity], list[str]]:
    entities: list[Entity] = []
    unresolved: list[str] = []
    for fav in favorites:
        ent = client.resolve(fav)
        if ent is None:
            unresolved.append(fav)
            trace.append(TraceStep("resolve_entity", f"'{fav}': no match in the taste graph", ok=False))
        else:
            entities.append(ent)
            trace.append(TraceStep("resolve_entity", f"'{fav}' -> {ent.name} ({ent.type or 'unknown type'})"))
    return entities, unresolved


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
    exclude_ids = exclude_ids or set()
    exclude_names = {n.lower() for n in (exclude_names or set())}
    trace: list[TraceStep] = [
        TraceStep("diagnose_slump", f"{slump.label}" + (" (prefers shorter reads)" if slump.prefers_short else ""))
    ]

    mode = "mock" if qloo.is_mock else "live"
    try:
        entities, unresolved = _resolve_all(qloo, favs, trace)
        candidates = qloo.book_recs(entities) if entities else []
    except QlooUnavailable as exc:
        trace.append(TraceStep("taste_graph", f"unavailable ({exc}); using clearly labeled mock data", ok=False))
        fallback = QlooClient(api_key="")
        trace_fb: list[TraceStep] = []
        entities, unresolved = _resolve_all(fallback, favs, trace_fb)
        trace.extend(trace_fb)
        candidates = fallback.book_recs(entities) if entities else []
        mode = "live-fallback-mock"

    if entities:
        trace.append(TraceStep("book_recommendations", f"{len(candidates)} candidate books from {len(entities)} favorites"))

    candidates = [c for c in candidates if c.id not in exclude_ids and c.name.lower() not in exclude_names]
    if exclude_ids or exclude_names:
        trace.append(TraceStep("exclusions", f"{len(candidates)} candidates left after your feedback"))

    if not candidates:
        return AgentResult(
            slump_key=slump.key,
            slump_label=slump.label,
            way_back=slump.way_back,
            picks=[],
            unresolved=unresolved,
            trace=trace,
            mode=mode,
            llm_mode=llm.mode,
        )

    by_id = {c.id: c for c in candidates}
    llm_mode = llm.mode
    try:
        raw_picks = llm.select_picks(candidates, slump, favs, minutes)
        trace.append(TraceStep("select_picks", f"{llm.mode} model chose from {len(candidates)} candidates"))
    except LLMError as exc:
        trace.append(TraceStep("select_picks", f"{llm.mode} model failed ({exc}); using mock selection", ok=False))
        raw_picks = MockLLM().select_picks(candidates, slump, favs, minutes)
        llm_mode = "mock"

    picks: list[Pick] = []
    used: set[str] = set()
    dropped = 0
    for p in raw_picks:
        cand = by_id.get(p.get("id", ""))
        if cand is None or cand.id in used:
            dropped += 1
            continue
        used.add(cand.id)
        picks.append(Pick(candidate=cand, why=p.get("why") or _template_why(cand)))
    if dropped:
        trace.append(TraceStep("grounding_check", f"dropped {dropped} pick(s) not found in taste-graph results", ok=False))
    else:
        trace.append(TraceStep("grounding_check", "all picks come from taste-graph results"))

    for cand in candidates:  # top up if the model returned fewer than 3 valid picks
        if len(picks) >= 3:
            break
        if cand.id not in used:
            used.add(cand.id)
            picks.append(Pick(candidate=cand, why=_template_why(cand)))

    return AgentResult(
        slump_key=slump.key,
        slump_label=slump.label,
        way_back=slump.way_back,
        picks=picks[:3],
        unresolved=unresolved,
        trace=trace,
        mode=mode,
        llm_mode=llm_mode,
    )


def build_llm(gemini_key: str, model: str):
    return GeminiLLM(gemini_key, model) if gemini_key else MockLLM()
