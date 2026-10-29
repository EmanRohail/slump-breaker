"""Second probe: how search results look for typical favorites, whether book
themes (tags) can come back, and what the rate limits look like.

PowerShell:
    $env:QLOO_API_KEY = "paste-your-key-here"
    python tools/probe_qloo_2.py

Saves qloo_probe2_output.txt (your key is never written). Paste it into the chat.
"""
from __future__ import annotations

import json
import os
import sys

import requests

BASE = os.environ.get("QLOO_BASE_URL", "https://hackathon.api.qloo.com").rstrip("/")
KEY = os.environ.get("QLOO_API_KEY", "")
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def get(path: str, params: dict):
    resp = requests.get(f"{BASE}{path}", params=params, headers={"X-Api-Key": KEY}, timeout=30)
    return resp


def main() -> int:
    if not KEY:
        print("Set QLOO_API_KEY first (see the top of this file).")
        return 1

    say("A. What each search returns (take=10): name | type | popularity | note")
    ids = {}
    for q in ["Hades", "Hozier", "Stranger Things", "Dune", "Studio Ghibli", "Animal Crossing", "Taylor Swift"]:
        r = get("/search", {"query": q, "take": 10})
        say(f"\n{q}: HTTP {r.status_code}")
        if r.status_code != 200:
            continue
        results = r.json().get("results", [])
        say(f"  {len(results)} results")
        for e in results:
            kind = (e.get("types") or ["?"])[0].rsplit(":", 1)[-1]
            say(f"  - {e.get('name')} | {kind} | pop={round(e.get('popularity') or 0, 3)} | {e.get('disambiguation')}")
            if kind == "videogame" and q not in ids:
                ids[q] = e.get("entity_id")
            if kind in ("tv_show", "movie") and q not in ids:
                ids[q] = e.get("entity_id")

    say("\nB. Does the `types` filter work on search?")
    r = get("/search", {"query": "Hades", "types": "urn:entity:videogame", "take": 5})
    say(f"HTTP {r.status_code}")
    if r.status_code == 200:
        for e in r.json().get("results", []):
            say(f"  - {e.get('name')} | {(e.get('types') or ['?'])[0]}")

    seed = ids.get("Hades") or ids.get("Stranger Things")
    say(f"\nC. Insights with explainability for seed {seed}")
    if seed:
        r = get("/v2/insights", {"filter.type": "urn:entity:book", "signal.interests.entities": seed,
                                 "take": 3, "feature.explainability": "true"})
        say(f"HTTP {r.status_code}")
        if r.status_code == 200:
            ents = r.json().get("results", {}).get("entities", [])
            for e in ents[:2]:
                say(f"  {e.get('name')}: keys={sorted(e.keys())}")
                say("  query = " + json.dumps(e.get("query"), indent=1)[:1500])
        else:
            say(r.text[:500])
    say("\nD. Rate-limit headers on the last response:")
    for k, v in r.headers.items() if seed else []:
        if "limit" in k.lower() or "retry" in k.lower() or "remaining" in k.lower():
            say(f"  {k}: {v}")

    with open("qloo_probe2_output.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nSaved to qloo_probe2_output.txt. Paste its contents into the chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
