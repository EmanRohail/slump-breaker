"""Email + password accounts through Supabase Auth (GoTrue REST).

Optional: with no SUPABASE_URL / SUPABASE_ANON_KEY the app offers guest use only.
The anon key is a public client key by design; it is still read from settings,
not hard-coded. Passwords go straight to Supabase and are never stored or
logged here. The saved profile (role, business name, channel, tone) lives in the
account's user_metadata, so there is no database of our own to run.

Tokens are kept in the Streamlit session only: a browser refresh signs you out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
PROFILE_KEYS = ("role", "btype", "bname", "channel", "tone", "name")


class AuthError(Exception):
    """A problem worth showing to the person, in plain words."""


@dataclass
class AuthUser:
    id: str
    email: str
    access_token: str = ""
    profile: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.profile.get("name") or self.email.split("@")[0]


def configured_url(url: str) -> str:
    """Only https (or localhost for development) is accepted."""
    p = urlparse((url or "").strip())
    if p.scheme == "https" and p.netloc:
        return f"https://{p.netloc}"
    if p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1"):
        return f"http://{p.netloc}"
    return ""


class AuthClient:
    def __init__(self, url: str, anon_key: str, session: Optional[requests.Session] = None) -> None:
        self.url = configured_url(url)
        self.key = (anon_key or "").strip()
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    # ----------------------------------------------------------------- actions

    def sign_up(self, email: str, password: str, name: str = "") -> tuple[Optional[AuthUser], str]:
        """Returns (user, note). user is None when the email must be confirmed first."""
        email = self._check(email, password, new=True)
        data = {"name": name.strip()[:60]} if name.strip() else {}
        body = self._post("/auth/v1/signup", {"email": email, "password": password, "data": data})
        if body.get("access_token"):
            return self._user(body), ""
        return None, "Check your email for a confirmation link, then sign in."

    def sign_in(self, email: str, password: str) -> AuthUser:
        email = self._check(email, password, new=False)
        body = self._post("/auth/v1/token?grant_type=password", {"email": email, "password": password})
        if not body.get("access_token"):
            raise AuthError("Sign in failed. Check your email and password.")
        return self._user(body)

    def save_profile(self, user: AuthUser, profile: dict) -> None:
        clean = {k: str(profile[k])[:80] for k in PROFILE_KEYS if k in profile}
        body = self._send("PUT", "/auth/v1/user", {"data": clean}, token=user.access_token)
        meta = (body or {}).get("user_metadata")
        user.profile = dict(meta) if isinstance(meta, dict) else {**user.profile, **clean}

    def sign_out(self, user: AuthUser) -> None:
        try:
            self._send("POST", "/auth/v1/logout", {}, token=user.access_token)
        except AuthError:
            pass  # the local session ends either way

    def reset_password(self, email: str) -> None:
        email = (email or "").strip().lower()
        if not _EMAIL.match(email):
            raise AuthError("Enter a valid email address.")
        self._post("/auth/v1/recover", {"email": email})

    # ---------------------------------------------------------------- plumbing

    def _check(self, email: str, password: str, new: bool) -> str:
        if not self.enabled:
            raise AuthError("Accounts are not set up on this deployment.")
        email = (email or "").strip().lower()
        if not _EMAIL.match(email):
            raise AuthError("Enter a valid email address.")
        if not password:
            raise AuthError("Enter your password.")
        if new and len(password) < MIN_PASSWORD:
            raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
        return email

    def _post(self, path: str, payload: dict) -> dict:
        return self._send("POST", path, payload)

    def _send(self, method: str, path: str, payload: dict, token: str = "") -> dict:
        if not self.enabled:
            raise AuthError("Accounts are not set up on this deployment.")
        headers = {"apikey": self.key, "Authorization": f"Bearer {token or self.key}", "Content-Type": "application/json"}
        try:
            resp = self.session.request(method, self.url + path, json=payload, headers=headers, timeout=15)
        except requests.RequestException:
            raise AuthError("Could not reach the sign-in service. Try again in a moment.") from None
        try:
            body = resp.json() if resp.content else {}
        except ValueError:
            body = {}
        if resp.status_code >= 400:
            raise AuthError(_friendly(resp.status_code, body))
        return body if isinstance(body, dict) else {}

    def _user(self, body: dict) -> AuthUser:
        u = body.get("user") or body
        meta = u.get("user_metadata")
        return AuthUser(
            id=str(u.get("id", "")),
            email=str(u.get("email", "")),
            access_token=str(body.get("access_token", "")),
            profile=dict(meta) if isinstance(meta, dict) else {},
        )


def _friendly(status: int, body: dict) -> str:
    raw = str(body.get("msg") or body.get("error_description") or body.get("message") or "").lower()
    if status == 429 or "rate limit" in raw:
        return "Too many attempts. Wait a minute and try again."
    if "invalid login" in raw or "invalid_credentials" in raw or "invalid credentials" in raw:
        return "Wrong email or password."
    if "already" in raw and ("registered" in raw or "exists" in raw):
        return "That email already has an account. Try signing in."
    if "not confirmed" in raw:
        return "Confirm your email first (check your inbox), then sign in."
    if "password" in raw and ("weak" in raw or "short" in raw or "least" in raw):
        return "That password is too weak. Use at least 8 characters."
    if status >= 500:
        return "The sign-in service had a problem. Try again shortly."
    return "That didn't work. Check what you entered and try again."
