"""Check your Gemini key, one step at a time. Run from the project folder:  python tools/check_gemini.py

Reads GEMINI_API_KEY from .streamlit/secrets.toml (or the environment). The key is never printed.
"""
import os
import sys
import tomllib
from pathlib import Path

import requests

key = os.environ.get("GEMINI_API_KEY", "")
secrets = Path(".streamlit/secrets.toml")
if not key and secrets.exists():
    key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("GEMINI_API_KEY", "")
if not key:
    sys.exit("No GEMINI_API_KEY found in .streamlit/secrets.toml. Add it first.")
print(f"Key found ({len(key)} characters, starts with {key[:3]}...).")
headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
base = "https://generativelanguage.googleapis.com/v1beta/models"

try:
    r = requests.get(base, headers=headers, params={"pageSize": 100}, timeout=20)
except requests.RequestException as exc:
    sys.exit(f"Could not reach Google at all ({exc.__class__.__name__}). Check your internet or VPN.")
print(f"1. List models: HTTP {r.status_code}")
usable = []
if r.status_code == 200:
    for m in r.json().get("models", []):
        n = m["name"].removeprefix("models/")
        if ("generateContent" in (m.get("supportedGenerationMethods") or []) and "flash" in n
                and not any(x in n for x in ("tts", "image", "live", "audio", "embedding"))):
            usable.append(n)
    print("   flash models your key can use:", ", ".join(usable[:8]) or "none")
else:
    print("   ", r.text[:300].replace("\n", " "))

ok = []
for model in dict.fromkeys(["gemini-flash-latest", *usable[:5]]):
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}]}
    try:
        r = requests.post(f"{base}/{model}:generateContent", headers=headers, json=body, timeout=30)
    except requests.RequestException as exc:
        print(f"2. {model}: network error {exc.__class__.__name__}")
        continue
    note = "works" if r.status_code == 200 else r.text[:200].replace("\n", " ")
    print(f"2. {model}: HTTP {r.status_code} - {note}")
    if r.status_code == 200:
        ok.append(model)

print()
if ok:
    print("RESULT: your key works. Models that answered:", ", ".join(ok))
else:
    print("RESULT: no model answered. 400/403 = the key is wrong or not allowed; 429 = free quota used up for now;")
    print("503 = Google is overloaded, try again in a few minutes; 404 = that model name is gone for your key.")
