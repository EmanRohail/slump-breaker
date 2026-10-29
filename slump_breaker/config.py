"""Settings come from environment variables first, then Streamlit secrets.

No key means mock mode, and the UI labels it clearly.
"""
from __future__ import annotations

import os

DEFAULT_QLOO_BASE_URL = "https://hackathon.api.qloo.com"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"


def _from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st

        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def get_setting(name: str, default: str = "") -> str:
    value = os.environ.get(name, "") or _from_streamlit_secrets(name)
    return value or default


def qloo_api_key() -> str:
    return get_setting("QLOO_API_KEY")


def qloo_base_url() -> str:
    return get_setting("QLOO_BASE_URL", DEFAULT_QLOO_BASE_URL)


def gemini_api_key() -> str:
    return get_setting("GEMINI_API_KEY")


def gemini_model() -> str:
    return get_setting("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def supabase_url() -> str:
    return get_setting("SUPABASE_URL")


def supabase_anon_key() -> str:
    return get_setting("SUPABASE_ANON_KEY")


def max_readers() -> int:
    """Readers per batch. The real ceilings are Qloo's request quota and run time, not file size."""
    try:
        return max(1, min(int(get_setting("MAX_READERS", "200")), 1000))
    except ValueError:
        return 200
