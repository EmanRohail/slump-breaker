"""The five slump types and the low-effort way back in for each."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlumpType:
    key: str
    label: str
    way_back: str
    prefers_short: bool = False


SLUMP_TYPES: dict[str, SlumpType] = {
    "taste_mismatch": SlumpType(
        "taste_mismatch",
        "The last book wasn't my taste",
        "Skip the genre you just tried. Start from something you already love on screen or in music "
        "and read the book that sits closest to it. Give it 20 pages, with full permission to stop.",
    ),
    "burnout": SlumpType(
        "burnout",
        "Burnout",
        "Go small and gentle: a short, low-stakes book you can finish in a couple of sittings. "
        "No goals, just 10 minutes a day.",
        prefers_short=True,
    ),
    "too_long": SlumpType(
        "too_long",
        "Too long or dense",
        "Pick the shortest strong match first, so your first finish takes days, not weeks. "
        "Save the big ones for after the habit is back.",
        prefers_short=True,
    ),
    "lost_habit": SlumpType(
        "lost_habit",
        "Lost the habit",
        "Rebuild the habit before the book: same time, same spot, 10 minutes, "
        "with a fast-moving pick that makes the next page easy.",
    ),
    "nothing_new": SlumpType(
        "nothing_new",
        "Nothing feels new",
        "Break the pattern: take the pick that connects two or more of your favorites "
        "in a way you would not have searched for.",
    ),
}

REASON_OPTIONS: list[tuple[str, str]] = [(t.key, t.label) for t in SLUMP_TYPES.values()]


def parse_reason(text: str) -> tuple[str, bool]:
    """Turn free text from a CSV into a slump key. Returns (key, was_guessed).

    A blank or unrecognized reason falls back to 'lost_habit' and is flagged as a
    guess, so the UI can say so instead of presenting it as fact.
    """
    cleaned = (text or "").strip().lower().replace("-", " ").replace("_", " ")
    if cleaned:
        for key, slump in SLUMP_TYPES.items():
            if cleaned in (key.replace("_", " "), slump.label.lower()):
                return key, False
        aliases = {
            "taste": "taste_mismatch", "mismatch": "taste_mismatch", "not my taste": "taste_mismatch",
            "tired": "burnout", "burned out": "burnout", "burnt out": "burnout",
            "long": "too_long", "dense": "too_long", "too long or dense": "too_long",
            "habit": "lost_habit", "busy": "lost_habit",
            "nothing new": "nothing_new", "bored": "nothing_new", "stale": "nothing_new",
        }
        if cleaned in aliases:
            return aliases[cleaned], False
    return "lost_habit", True


def get_slump(key: str) -> SlumpType:
    if key not in SLUMP_TYPES:
        raise ValueError(f"Unknown slump type: {key!r}")
    return SLUMP_TYPES[key]
