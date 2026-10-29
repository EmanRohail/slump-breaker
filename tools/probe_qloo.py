"""Run this on YOUR computer to see what Qloo really returns.

It sends a few read-only requests, prints the shape of each answer, and saves
everything to qloo_probe_output.txt. Your key is read from an environment
variable and is never printed or written to the file.

PowerShell:
    $env:QLOO_API_KEY = "paste-your-key-here"
    python tools/probe_qloo.py

Then paste the contents of qloo_probe_output.txt back into the chat.
"""
from __future__ import annotations

import json
import os
import sys

import requests

BASE = os.environ.get("QLOO_BASE_URL", "https://hackathon.api.qloo.com").rstrip("/")
KEY = os.environ.get("QLOO_API_KEY", "")
OUT = "qloo_probe_output.txt"
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def shape(obj, depth: int = 0, max_depth: int = 5) -> str:
    pad = "  " * depth
    if depth > max_depth:
        return pad + "..."
    if isinstance(obj, dict):
        rows = [f"{pad}{{"]
        for k, v in list(obj.items())[:25]:
            inner = shape(v, depth + 1, max_depth).lstrip()
            rows.append(f"{pad}  {k}: {inner}")
        rows.append(f"{pad}}}")
        return "\n".join(rows)
    if isinstance(obj, list):
        if not obj:
            return pad + "[] (empty)"
        return f"{pad}list[{len(obj)}] of " + shape(obj[0], depth, max_depth).lstrip()
    text = json.dumps(obj)
    return pad + (text[:60] + "..." if len(text) > 60 else text)


def call(label: str, path: str, params: dict):
    say("=" * 70)
    say(f"{label}: GET {path}  params={params}")
    try:
        resp = requests.get(f"{BASE}{path}", params=params, headers={"X-Api-Key": KEY}, timeout=30)
    except requests.RequestException as exc:
        say(f"NETWORK ERROR: {exc.__class__.__name__}")
        return None
    say(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        say("Not JSON. First 500 chars:\n" + resp.text[:500])
        return None
    say("SHAPE:\n" + shape(data))
    raw = json.dumps(data, indent=1)
    say("RAW (first 2500 chars):\n" + raw[:2500])
    return data


def find_first_id(obj):
    if isinstance(obj, dict):
        for k in ("entity_id", "id"):
            if isinstance(obj.get(k), str):
                return obj[k]
        for v in obj.values():
            found = find_first_id(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_first_id(v)
            if found:
                return found
    return None


def main() -> int:
    if not KEY:
        print("Set QLOO_API_KEY first (see the top of this file).")
        return 1
    say(f"Base URL: {BASE}")
    hades = call("1. Search a game", "/search", {"query": "Hades", "take": 3})
    call("2. Search a band", "/search", {"query": "Hozier", "take": 3})
    call("3. Search a book (checks book support)", "/search", {"query": "Circe", "take": 3})
    seed = find_first_id(hades)
    say(f"\nSeed id used for insights: {seed}")
    if seed:
        call(
            "4. Book recommendations for the seed",
            "/v2/insights",
            {"filter.type": "urn:entity:book", "signal.interests.entities": seed, "take": 5},
        )
    else:
        say("No id found in search result, so insights was skipped.")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved to {OUT}. Paste that file's contents into the chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
