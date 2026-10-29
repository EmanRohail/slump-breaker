"""Business view: turn lapsed readers into personal win-back messages.

The business gives us readers it already knows have lapsed (name, favorites,
optionally why). We pick a book per reader from the taste graph and write a
message. This is a proposed layer on top of taste data: it makes no claims about
churn or conversion, and nothing is stored.
"""
from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from . import agent, config, mock_data
from .llm import LLMError, MockLLM
from .models import AgentResult, spoken
from .slump_types import SLUMP_TYPES, parse_reason

MAX_ROWS = config.max_readers()  # default 200; the real limits are Qloo quota and run time
MAX_CSV_CHARS = 5_000_000  # about 50,000 readers of text, far above MAX_ROWS
CHANNELS = ["email", "push", "sms"]
TONES = ["warm", "playful", "brief"]
_SPLIT = re.compile(r"[;|]")


@dataclass
class ReaderRow:
    name: str
    favorites: list[str]
    reason_key: str = "lost_habit"
    reason_guessed: bool = True


@dataclass
class WinBackPlan:
    slump_label: str
    pick_title: str
    pick_author: str
    message: str
    evidence: list[str]
    message_source: str  # "gemini" or "mock"
    alternates: list[tuple[str, str]] = field(default_factory=list)
    reason_guessed: bool = False


@dataclass
class BatchItem:
    row: ReaderRow
    result: Optional[AgentResult] = None
    plan: Optional[WinBackPlan] = None
    error: str = ""


# ------------------------------------------------------------------- CSV input


def parse_csv(text: str) -> tuple[list[ReaderRow], list[str]]:
    """Read name,favorites,reason (reason optional). Returns (rows, problems)."""
    problems: list[str] = []
    text = (text or "").lstrip("﻿")
    if not text.strip():
        return [], ["The file is empty."]
    if len(text) > MAX_CSV_CHARS:
        return [], [f"The file is too large (limit {MAX_CSV_CHARS:,} characters)."]
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}
    if "name" not in fields or "favorites" not in fields:
        return [], ["The first row must be a header with at least: name, favorites (reason is optional)."]
    rows: list[ReaderRow] = []
    for i, raw in enumerate(reader, start=2):
        name = (raw.get(fields["name"]) or "").strip()
        parts = [x for x in _SPLIT.split(raw.get(fields["favorites"]) or "") if x.strip()]
        favs = agent.clean_favorites(parts)
        if len({x.strip().lower() for x in parts}) > agent.MAX_FAVORITES:
            problems.append(f"Row {i}: only the first {agent.MAX_FAVORITES} favorites are used.")
        if not name and not favs:
            continue
        if not name or not favs:
            problems.append(f"Row {i}: needs both a name and at least one favorite, so it was skipped.")
            continue
        if len(rows) >= MAX_ROWS:
            problems.append(f"Only the first {MAX_ROWS} readers are used per run; the rest were ignored. Split the file to do more.")
            break
        reason_text = raw.get(fields["reason"]) if "reason" in fields else ""
        key, guessed = parse_reason(reason_text or "")
        rows.append(ReaderRow(name=name, favorites=favs, reason_key=key, reason_guessed=guessed))
    if not rows and not problems:
        problems.append("No readers found in the file.")
    return rows, problems


def template_csv() -> str:
    sample = [
        ("Maya", "Stranger Things; Grand Theft Auto V; Taylor Swift", "taste mismatch"),
        ("Omar", "Dune; Interstellar; Hans Zimmer", "too long"),
        ("Sana", "Studio Ghibli; Minecraft; Harry Potter", ""),
    ]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["name", "favorites", "reason"])
    writer.writerows(sample)
    return out.getvalue()


def sample_rows() -> list[ReaderRow]:
    rows, _ = parse_csv(template_csv())
    return rows


# ------------------------------------------------------------------ the plan


def build_plan(
    result: AgentResult,
    reader_name: str,
    llm,
    channel: str = "email",
    tone: str = "warm",
    business: str = "",
    reason_guessed: bool = False,
) -> Optional[WinBackPlan]:
    if not result.picks:
        return None
    top = result.picks[0].candidate
    author = top.author or "an author you may know"
    source = llm.mode
    try:
        message = llm.write_message(reader_name, top.name, author, spoken(top.fits, result.genre_of), result.way_back, channel, tone, business)
    except LLMError:
        message = MockLLM().write_message(reader_name, top.name, author, spoken(top.fits, result.genre_of), result.way_back, channel, tone, business)
        source = "mock"
    g = result.genre_of
    evidence = []
    for label in spoken(top.seeds, g):
        via = [s for s in top.seeds if g.get(s, s) == label and s != label]
        evidence.append(f"Linked to the reader's favorite: {label}" + (f" (through {', '.join(via)})" if via else ""))
    if top.blend:
        evidence.append("Fits their whole mix together: " + ", ".join(spoken(top.blend, g)))
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
        alternates=[(p.candidate.name, p.candidate.author) for p in result.picks[1:3]],
        reason_guessed=reason_guessed,
    )


def estimate_requests(rows: Iterable[ReaderRow]) -> int:
    """Rough Qloo request count. Results are cached, so each distinct favorite costs about 3 calls once."""
    from .qloo import norm_title

    distinct = {norm_title(f) for r in rows for f in r.favorites}
    return len(distinct) * 3


def run_batch(
    rows: Iterable[ReaderRow],
    qloo,
    llm,
    channel: str = "email",
    tone: str = "warm",
    business: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    fast: bool = False,
    workers: int = 3,
) -> list[BatchItem]:
    """Plan every reader, a few at a time. `fast` uses the offline planner and template text (no model calls),
    which keeps big lists inside free-tier model limits. If the model fails, it is paused for a short while
    (see GeminiLLM), so later readers fall back to the fixed plan at once instead of each waiting on retries."""
    rows = list(rows)
    items = [BatchItem(row=r) for r in rows]

    def work(i: int) -> None:
        row, item = rows[i], items[i]
        use = MockLLM() if fast else llm
        try:  # one bad row must not stop the batch
            item.result = agent.run(row.favorites, row.reason_key, None, qloo, use)
            item.plan = build_plan(item.result, row.name, use, channel, tone, business, row.reason_guessed)
        except Exception as exc:  # noqa: BLE001
            item.error = f"{exc.__class__.__name__}: {exc}"

    if len(rows) <= 1 or workers <= 1:
        for i in range(len(rows)):
            work(i)
            if progress:
                progress(i + 1, len(rows))
        return items
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i) for i in range(len(rows))]
        for done, fut in enumerate(as_completed(futures), start=1):
            fut.result()
            if progress:
                progress(done, len(rows))  # called from the calling thread, so the UI can update
    return items


# ----------------------------------------------------------------- CSV output


def _safe_cell(value: str) -> str:
    """Stop spreadsheet apps from running a cell that starts like a formula."""
    value = str(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def export_csv(items: Iterable[BatchItem], channel: str = "email") -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        ["name", "likely_reason", "reason_was_guessed", "pick", "author", "alternates",
         "linked_favorites", "channel", "message", "data_source"]
    )
    for it in items:
        label = SLUMP_TYPES[it.row.reason_key].label
        if it.plan and it.result:
            top = it.result.picks[0].candidate
            writer.writerow([_safe_cell(v) for v in [
                it.row.name, label, "yes" if it.row.reason_guessed else "no",
                it.plan.pick_title, it.plan.pick_author,
                "; ".join(f"{t} by {a}" for t, a in it.plan.alternates),
                "; ".join(spoken(top.fits, it.result.genre_of)), channel, it.plan.message,
                "mock fixture" if top.source == "mock" else "Qloo taste graph",
            ]])
        else:
            writer.writerow([_safe_cell(v) for v in [
                it.row.name, label, "yes" if it.row.reason_guessed else "no",
                "", "", "", "", channel, it.error or "No match found for these favorites.", "",
            ]])
    return out.getvalue()
