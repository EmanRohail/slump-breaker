"""Slump Breaker: Streamlit app. Run with: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from slump_breaker import agent, auth, business, config, mock_data, theme
from slump_breaker.llm import LLMError
from slump_breaker.models import spoken, AgentResult, blend_problem, fallback_reason
from slump_breaker.qloo import TROPES, QlooClient
from slump_breaker.slump_types import REASON_OPTIONS, SLUMP_TYPES

st.set_page_config(page_title="Slump Breaker", page_icon="📖", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


@st.cache_resource
def get_qloo() -> QlooClient:
    return QlooClient(api_key=config.qloo_api_key(), base_url=config.qloo_base_url())


@st.cache_resource
def get_llm():
    return agent.build_llm(config.gemini_api_key(), config.gemini_model())


@st.cache_resource
def get_auth() -> auth.AuthClient:
    return auth.AuthClient(config.supabase_url(), config.supabase_anon_key())


qloo = get_qloo()
llm = get_llm()
accounts = get_auth()
ss = st.session_state

BIZ_TYPES = ["Reading app", "Book box or subscription", "Publisher", "Library or bookstore", "Something else"]
REASON_LABELS = {k: v for k, v in REASON_OPTIONS}
PAGES = ["Business", "Reader", "Why Qloo?"]
PAGE_LABELS = {"Business": "Business view", "Reader": "Reader view", "Why Qloo?": "Why Qloo?"}

ss.setdefault("page", "Business")
ss.setdefault("welcomed", False)
ss.setdefault("user", None)
ss.setdefault("history", [])
ss.setdefault("last_page", None)
ss.setdefault("why_result", None)
ss.setdefault("onboarded", False)
ss.setdefault("step", 1)
ss.setdefault("role", "business")
ss.setdefault("profile", {})
ss.setdefault("rows", [])
ss.setdefault("batch", None)
ss.setdefault("r_inputs", None)
ss.setdefault("r_result", None)
ss.setdefault("excl_ids", set())
ss.setdefault("excl_names", set())
ss.setdefault("worked", set())
ss.setdefault("plain_cache", {})


def esc_text(text: str) -> str:
    return theme.esc(text)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def chips() -> None:
    html(theme.status_chips(qloo.is_mock, llm.mode))


def demo_banner() -> None:
    """Say plainly when no Qloo key is loaded, because only a few titles work then."""
    if qloo.is_mock:
        st.warning(
            "**Demo mode: no Qloo key was found, so results use a small sample set.** Only these titles work: "
            + ", ".join(mock_data.supported_titles())
            + ". To use real Qloo data, put your key in `.streamlit/secrets.toml` as `QLOO_API_KEY = \"...\"` "
            "(or set the QLOO_API_KEY environment variable) and restart the app."
        )


MAX_DAILY_MINUTES = 10 * 60


def time_picker(prefix: str, default_minutes: int = 10) -> int:
    """Type hours and minutes (0-10 h, 0-59 min). Returns total minutes; 0 means no preference."""
    default_minutes = max(0, min(int(default_minutes or 0), MAX_DAILY_MINUTES))
    h, m = st.columns(2)
    hours = h.number_input("Hours", 0, 10, default_minutes // 60, step=1, key=f"{prefix}_h")
    mins = m.number_input("Minutes", 0, 59, default_minutes % 60, step=5, key=f"{prefix}_m")
    total = int(hours) * 60 + int(mins)
    if total > MAX_DAILY_MINUTES:
        st.warning("That is more than 10 hours a day, so we will use 10 hours.")
        total = MAX_DAILY_MINUTES
    st.caption(f"About {total} minutes a day." if total else "No daily time preference.")
    return total


# What the taste graph can use as an input, as the person sees it -> the hint word qloo.parse_hint understands.
FAV_TYPES = [  # book kinds first: they are the point of the app, and stay visible in a short dropdown
    ("Auto-detect", ""),
    ("Book", "book"),
    ("Book genre", "genre"),
    ("Trope", "trope"),
    ("TV show or series", "show"),
    ("Movie", "movie"),
    ("Video game", "game"),
    ("Music artist or band", "artist"),
]
_TYPE_LABEL_BY_KIND = {"tv_show": "TV show or series", "movie": "Movie", "videogame": "Video game",
                       "artist": "Music artist or band", "book": "Book", "genre": "Book genre", "trope": "Trope"}
_WORD_BY_LABEL = dict(FAV_TYPES)


def favorites_help() -> None:
    with st.expander("What can I enter?"):
        st.markdown(
            "**Add up to five things you love.** The more different they are, the better the match.\n\n"
            "**What works well**\n"
            "- TV shows and series, anime, movies, video games, music artists or bands, and books.\n"
            "- The official title, for example *Stranger Things*, *Grand Theft Auto V*, *BTS* or *Healer*. "
            "Pick its type so we search the right category.\n"
            "- A book genre such as *dark romance*, *romcom*, *romantasy* or *cozy mystery* (choose **Book genre**). "
            "We use the books Qloo files under that genre, and next to a title it narrows that title's picks to the genre.\n"
            "- A trope (choose **Trope**): " + ", ".join(f"*{t}*" for t in TROPES) + ". "
            "We start from a couple of books famous for it (the trace shows which).\n"
            "- A screen genre like *anime* or *kdrama* works as a starting point too: we pair it with a couple of "
            "very well-known titles of that kind (the trace shows which). A specific title always gives a sharper pick.\n\n"
            "**Good to know**\n"
            "- For an album or a single song, add the artist instead.\n"
            "- If we can't find a title, we tell you which one, so you can try another."
        )


def favorite_slots(prefix: str, preset: list[str] | None = None, show_help: bool = True) -> list[str]:
    """Exactly MAX_FAVORITES rows (title + type), so nobody can enter a sixth. Returns the filled ones, de-duplicated.

    The type is sent to Qloo as a search filter, which avoids mix-ups like a game matched to an album.
    `preset` fills the rows when it changes (a demo reader, a finished run); edits in between are kept.
    """
    from slump_breaker.qloo import parse_hint

    n = agent.MAX_FAVORITES
    if preset is not None and (ss.get(f"{prefix}_sig") != tuple(preset) or f"{prefix}_0" not in ss):
        ss[f"{prefix}_sig"] = tuple(preset)
        for i in range(n):
            base, kind = parse_hint(preset[i]) if i < len(preset) else ("", None)
            ss[f"{prefix}_{i}"] = base
            ss[f"{prefix}_t{i}"] = _TYPE_LABEL_BY_KIND.get(kind or "", "Auto-detect")
    placeholders = ["Grand Theft Auto V", "Stranger Things", "BTS", "Healer", "Harry Potter"]
    labels = [f for f, _ in FAV_TYPES]
    out: list[str] = []
    for i in range(n):
        left, right = st.columns([3, 2])
        title = left.text_input(f"Favorite {i + 1}" + (" (required)" if i == 0 else " (optional)"),
                                key=f"{prefix}_{i}", placeholder=placeholders[i])
        kind = right.selectbox("Type", labels, key=f"{prefix}_t{i}")
        parts = [p.strip()[:80] for p in title.split(";") if p.strip()]  # "A; B" in one box still works
        word = _WORD_BY_LABEL[kind]
        for p in parts:
            out.append(f"{p} ({word})" if word and len(parts) == 1 and "(" not in p else p)
    filled = agent.clean_favorites(out)
    st.caption(f"{len(filled)} of {n} added.")
    if show_help:
        favorites_help()
    return filled


def match_warnings(result: AgentResult) -> None:
    """Flag matches that are not the title the person typed (for example "sombr" matched to "Sombra")."""
    if result.uncertain:
        st.warning("Please check these matches, they are not an exact name: " + " · ".join(result.uncertain)
                   + ". Qloo may not know the one you meant. Use the full official name, and pick its type.")


def reader_editor(row: business.ReaderRow, key: str, rerun) -> None:
    """Change a reader's name, favorites or reason after they were added. `rerun` redoes the plan if one exists."""
    from slump_breaker.slump_types import SLUMP_TYPES as _ST

    with st.form(f"{key}_form"):  # nothing reruns until Save, so the box stays open while typing
        name = st.text_input("Reader's first name", value=row.name, key=f"{key}_name")
        favs = favorite_slots(f"{key}_fav", list(row.favorites), show_help=False)
        labels = ["I'm not sure"] + [t.label for t in _ST.values()]
        now = "I'm not sure" if row.reason_guessed else _ST[row.reason_key].label
        reason = st.selectbox("Why did they stop? (optional)", labels, index=labels.index(now), key=f"{key}_reason")
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        if not name.strip() or not favs:
            st.error("A reader needs a name and at least one favorite.")
            return
        row.name, row.favorites = name.strip(), favs
        row.reason_key, row.reason_guessed = business.parse_reason("" if reason == "I'm not sure" else reason)
        rerun(row)
        st.rerun()


def skipped_notes(result: AgentResult) -> None:
    """Say which favorites Qloo refused to use and why, so a missing favorite is never a mystery."""
    for step in result.trace:
        if step.tool == "get_book_recs" and not step.ok:
            st.caption("⚠ " + step.detail)


def go_back() -> None:
    if ss.history:
        ss.page = ss.history.pop()
        ss.last_page = ss.page


def change_view() -> None:
    """Back to the first onboarding step, where the business/reader choice is made."""
    ss.onboarded, ss.step, ss.batch = False, 1, None


def sign_out() -> None:
    if ss.user:
        accounts.sign_out(ss.user)
    ss.user = None
    for k in ("welcomed", "onboarded"):
        ss[k] = False
    ss.step, ss.profile, ss.rows, ss.batch, ss.r_inputs, ss.r_result = 1, {}, [], None, None, None


def save_profile() -> None:
    if ss.user:
        try:
            accounts.save_profile(ss.user, {**ss.profile, "role": ss.role})
        except auth.AuthError:
            pass  # saving is a convenience; the session continues


def enter_app(user: auth.AuthUser) -> None:
    ss.user, ss.welcomed = user, True
    prof = user.profile
    if prof.get("role") in ("business", "reader"):
        ss.role = prof["role"]
        ss.profile = {k: prof[k] for k in ("btype", "bname", "channel", "tone") if prof.get(k)}
        ss.onboarded = True
        ss.page = "Business" if ss.role == "business" else "Reader"
        if ss.role == "business":
            ss.rows = []
        else:
            ss.r_inputs, ss.r_result = None, None  # a returning reader goes straight to the app: no setup questions
    else:
        ss.step = 1


# ======================================================================== welcome


def welcome() -> None:
    html(theme.brand())
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        html(theme.hero("Welcome to Slump Breaker", "Win lapsed readers back with books that match what they already love."))
        if accounts.enabled:
            tab_in, tab_up = st.tabs(["Sign in", "Create account"])
            with tab_in:
                with st.form("signin"):
                    email = st.text_input("Email", key="si_email")
                    pw = st.text_input("Password", type="password", key="si_pw")
                    if st.form_submit_button("Sign in", type="primary", width="stretch"):
                        try:
                            enter_app(accounts.sign_in(email, pw))
                            st.rerun()
                        except auth.AuthError as exc:
                            st.error(str(exc))
                with st.expander("Forgot your password?"):
                    fe = st.text_input("Email for the reset link", key="fp_email")
                    if st.button("Send reset link"):
                        try:
                            accounts.reset_password(fe)
                            st.success("If that email has an account, a reset link is on its way.")
                        except auth.AuthError as exc:
                            st.error(str(exc))
            with tab_up:
                with st.form("signup"):
                    nm = st.text_input("Your name (optional)", key="su_name")
                    email = st.text_input("Email", key="su_email")
                    pw = st.text_input(f"Password (at least {auth.MIN_PASSWORD} characters)", type="password", key="su_pw")
                    if st.form_submit_button("Create account", type="primary", width="stretch"):
                        try:
                            user, note = accounts.sign_up(email, pw, nm)
                            if user:
                                enter_app(user)
                                st.rerun()
                            else:
                                st.success(note)
                        except auth.AuthError as exc:
                            st.error(str(exc))
            st.markdown("<div class='note' style='text-align:center;margin:.6rem 0'>or</div>", unsafe_allow_html=True)
        else:
            st.info("Sign-in is switched off on this deployment (no account service is configured), so you can continue as a guest.")
        if st.button("Continue as guest", width="stretch", type="secondary" if accounts.enabled else "primary"):
            ss.welcomed = True
            st.rerun()
        st.caption("Guests are not saved anywhere. Uploaded reader lists are used in your session only and are never stored.")


# ===================================================================== onboarding


def onboarding() -> None:
    html(theme.brand())
    _, mid, _ = st.columns([1, 2.4, 1])
    with mid:
        step = ss.step
        who = f"signed in as {ss.user.email}" if ss.user else "using Slump Breaker as a guest"
        html(f'<div class="step">Step {step} of 4 · {esc_text(who)}</div>')
        st.progress(step / 4)
        role = ss.role
        profile = ss.profile

        if step == 1:
            html(theme.hero("How will you use Slump Breaker?", "This sets up the right workspace for you."))
            c1, c2 = st.columns(2)
            with c1:
                html('<div class="panel"><div class="lbl">Business</div><h3>I run a book business</h3>'
                     "<p>Turn readers who stopped coming back into personal win-back messages.</p></div>")
                if st.button("Continue as a business", type="primary", width="stretch"):
                    ss.role, ss.step = "business", 2
                    st.rerun()
            with c2:
                html('<div class="panel"><div class="lbl">Reader</div><h3>I\'m a reader</h3>'
                     "<p>Find the book that gets you reading again, based on what you already love.</p></div>")
                if st.button("Continue as a reader", width="stretch"):
                    ss.role, ss.step = "reader", 2
                    st.rerun()
            if not ss.user and st.button("← Back to sign in"):
                ss.welcomed = False
                st.rerun()
            return

        if role == "business":
            if step == 2:
                html(theme.hero("What do you run?", "So the messages sound like they come from you."))
                btype = st.radio("Type of business", BIZ_TYPES, index=BIZ_TYPES.index(profile.get("btype", BIZ_TYPES[0])))
                bname = st.text_input("Business name (optional)", value=profile.get("bname", ""))
                nxt = {"btype": btype, "bname": bname.strip()}
            elif step == 3:
                html(theme.hero("How do you reach readers?", "We write each message to fit the channel."))
                channel = st.radio("Channel", business.CHANNELS, horizontal=True,
                                   index=business.CHANNELS.index(profile.get("channel", "email")))
                tone = st.radio("Tone", business.TONES, horizontal=True,
                                index=business.TONES.index(profile.get("tone", "warm")))
                nxt = {"channel": channel, "tone": tone}
            else:
                html(theme.hero("Want to try it with sample readers?",
                                "Three made-up readers, so you can see a result right away."))
                st.caption("You can upload your own list or add readers one by one at any time. Nothing is stored.")
                nxt = {}
        else:
            if step == 2:
                html(theme.hero("What do you love?", "Up to 5 favorites: shows, games, music, films or books."))
                demo = st.selectbox("Or start from a demo reader", ["(none)"] + list(mock_data.DEMO_READERS))
                if demo == "(none)" and profile.get("favs_demo"):
                    profile["favs"] = ""  # picking "(none)" empties every box again
                    profile.pop("favs_demo", None)
                if demo != "(none)" and profile.get("favs_demo") != demo:
                    profile["favs"] = "\n".join(mock_data.DEMO_READERS[demo]["favorites"])
                    profile["reason"] = mock_data.DEMO_READERS[demo]["reason"]
                    profile["favs_demo"] = demo
                favs = favorite_slots("ob_fav", [f for f in profile.get("favs", "").splitlines() if f.strip()])
                nxt = {"favs": "\n".join(favs)}
            elif step == 3:
                html(theme.hero("What happened to your reading?", "Pick the closest one."))
                keys = list(REASON_LABELS)
                reason = st.radio("Reason", keys, format_func=lambda k: REASON_LABELS[k],
                                  index=keys.index(profile.get("reason", keys[0])), label_visibility="collapsed")
                nxt = {"reason": reason}
            else:
                html(theme.hero("How much time can you give it?", "A small daily amount is plenty."))
                nxt = {"minutes": time_picker("ob_time", profile.get("minutes", 10))}

        back, _, forward = st.columns([1, 2, 2])
        if back.button("Back"):
            ss.step -= 1
            st.rerun()
        last = step == 4
        if role == "business" and last:
            if forward.button("Load sample readers", type="primary", width="stretch"):
                _finish(profile, sample=True)
            if st.button("Start empty", width="stretch"):
                _finish(profile, sample=False)
        elif forward.button("Finish" if last else "Next", type="primary", width="stretch"):
            if role == "reader" and step == 2 and not agent.clean_favorites(nxt["favs"].splitlines()):
                st.error("Add at least one thing you love.")
            else:
                profile.update(nxt)
                if last:
                    _finish(profile)
                ss.step += 1
                st.rerun()


def _finish(profile: dict, sample: bool = False) -> None:
    ss.onboarded = True
    save_profile()
    if ss.role == "business":
        ss.page = "Business"
        ss.rows = business.sample_rows() if sample else []
    else:
        ss.page = "Reader"
        favs = agent.clean_favorites(profile.get("favs", "").splitlines())
        ss.r_inputs = (favs, profile.get("reason", "lost_habit"), profile.get("minutes", 10))
        run_reader()
    st.rerun()


# ======================================================================== business


def page_business() -> None:
    top, right = st.columns([3, 2])
    with top:
        html(theme.hero("Win back readers, thoughtfully.",
                        "Turn a lapsed reader's taste into a personal reason to come back."))
    with right:
        chips()
    demo_banner()
    p = ss.profile

    c1, c2, c3 = st.columns(3)
    channel = c1.selectbox("Channel", business.CHANNELS, index=business.CHANNELS.index(p.get("channel", "email")))
    tone = c2.selectbox("Tone", business.TONES, index=business.TONES.index(p.get("tone", "warm")))
    bname = c3.text_input("Business name", value=p.get("bname", ""))
    p.update({"channel": channel, "tone": tone, "bname": bname})

    def replan(row: business.ReaderRow) -> None:
        """After an edit, redo the plan for that one reader if a plan already exists."""
        for k, it in enumerate(ss.batch or []):
            if it.row is row:
                with st.spinner("Updating this reader's plan..."):
                    ss.batch[k] = business.run_batch([row], qloo, llm, channel, tone, bname.strip())[0]

    html('<div class="panel"><div class="lbl">Step 1</div><h3>Who has stopped reading?</h3>'
         '<p class="note">Add readers you already know have lapsed. Nothing you enter is stored.</p></div>')
    tab_one, tab_csv = st.tabs(["Add one reader", "Upload a list"])
    with tab_one:
        with st.form("one_reader", clear_on_submit=True):
            name = st.text_input("Reader's first name")
            fav_list = favorite_slots("br_fav", None)
            reason = st.selectbox("Why did they stop? (optional)", ["I'm not sure"] + [t.label for t in SLUMP_TYPES.values()])
            if st.form_submit_button("Add reader"):
                if not name.strip() or not fav_list:
                    st.error("Add a name and at least one favorite.")
                elif len(ss.rows) >= business.MAX_ROWS:
                    st.error(f"Up to {business.MAX_ROWS} readers per run.")
                else:
                    key, guessed = business.parse_reason("" if reason == "I'm not sure" else reason)
                    ss.rows.append(business.ReaderRow(name.strip(), fav_list, key, guessed))
                    st.rerun()
    with tab_csv:
        st.caption(f"Columns: name, favorites (separate with ;), reason (optional). Up to {business.MAX_ROWS} readers per run. "
                   "Files are read in memory and never saved.")
        up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if up is not None:
            rows, problems = business.parse_csv(up.getvalue().decode("utf-8-sig", errors="replace"))
            for msg in problems:
                st.warning(msg)
            if rows and st.button(f"Use the {len(rows)} readers from this file"):
                ss.rows = rows
                st.rerun()
        d1, d2 = st.columns(2)
        d1.download_button("Download template", business.template_csv(), "slump-breaker-template.csv", "text/csv")
        if d2.button("Load 3 sample readers"):
            ss.rows = business.sample_rows()
            st.rerun()

    if ss.rows:
        st.caption("Tick the box at the start of a row to select it, then remove it.")
        event = st.dataframe(
            [{"Reader": r.name, "Favorites": "; ".join(r.favorites),
              "Reason": SLUMP_TYPES[r.reason_key].label + (" (guess)" if r.reason_guessed else "")} for r in ss.rows],
            hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
            key=f"rows_table_{ss.get('rows_version', 0)}",
        )
        picked = [i for i in event.selection.rows if 0 <= i < len(ss.rows)]
        if picked and st.button(f"Remove {len(picked)} selected reader{'s' if len(picked) > 1 else ''}"):
            ss.rows = [r for i, r in enumerate(ss.rows) if i not in set(picked)]
            ss.batch = None
            ss.rows_version = ss.get("rows_version", 0) + 1  # a new key clears the old selection
            st.rerun()
        with st.expander("Edit a reader's name, favorites or reason"):
            ver = ss.get("rows_version", 0)
            k = st.selectbox("Which reader?", range(len(ss.rows)), key=f"edit_pick_{ver}",
                             format_func=lambda i: f"{i + 1} · {ss.rows[i].name}")
            reader_editor(ss.rows[k], f"ed1_{ver}_{k}", replan)
        big = len(ss.rows) > 20
        fast = False
        if big:
            fast = st.checkbox("Fast mode: template messages instead of AI-written ones (recommended for long lists)",
                               value=True, help="Free AI plans allow only a few requests a minute. Picks still come from Qloo.")
        est = business.estimate_requests(ss.rows)
        if not qloo.is_mock:
            st.caption(f"About {est} Qloo requests (repeated favorites are cached) · roughly "
                       f"{max(1, round(est * 0.25 / 60 + len(ss.rows) * (0.2 if fast else 2) / 60))} min.")
        a, b, _ = st.columns([2, 1, 3])
        go = a.button("Create win-back plans", type="primary", width="stretch")
        if b.button("Clear", width="stretch"):
            ss.rows, ss.batch = [], None
            st.rerun()
        if go:
            bar = st.progress(0.0, text="Asking the taste graph...")
            ss.batch = business.run_batch(
                ss.rows, qloo, llm, channel, tone, bname.strip(),
                progress=lambda i, n: bar.progress(i / n, text=f"Reader {i} of {n}"), fast=fast,
            )
            ss.batch_channel = channel
            bar.empty()
    else:
        st.info("Add a reader, upload a list, or load the sample readers to begin.")

    items = ss.batch
    if not items:
        return
    html('<div class="panel"><div class="lbl">Step 2</div><h3>Your win-back plans</h3></div>')
    made = [i for i in items if i.plan]
    m1, m2, m3 = st.columns(3)
    m1.metric("Readers", len(items))
    m2.metric("Plans created", len(made))
    sources = {i.result.mode for i in items if i.result}
    m3.metric("Data source", "Live Qloo" if sources == {"live"} else "Sample data" if sources else "-")
    if sources - {"live"}:
        why = next((fallback_reason(i.result) for i in items if i.result and fallback_reason(i.result)), "")
        if why:
            st.error("Qloo could not be reached, so these plans use the built-in sample data, which only knows a few titles. "
                     f"Reason: {why}")
        else:
            st.warning("Some or all of these plans use the built-in sample data, not real Qloo data.")
    st.dataframe(
        [{"Reader": i.row.name,
          "Likely reason": SLUMP_TYPES[i.row.reason_key].label + (" (guess)" if i.row.reason_guessed else ""),
          "Comeback pick": i.plan.pick_title if i.plan else "No match",
          "Author": i.plan.pick_author if i.plan else ""} for i in items],
        hide_index=True, width="stretch",
    )
    names = [f"{k + 1} · {i.row.name}" for k, i in enumerate(items)]
    html('<div class="panel"><div class="lbl">Step 3</div><h3>Pick a reader to see their plan</h3>'
         '<p class="note">Each reader has their own pick, message and evidence.</p></div>')
    pick_key = f"plan_pick_{id(items)}"  # a new batch gets a fresh picker
    if len(items) <= 8:
        sel = st.pills("Reader", range(len(items)), selection_mode="single", default=0, key=pick_key,
                       format_func=lambda k: names[k], label_visibility="collapsed")
        chosen = 0 if sel is None else sel
    else:
        chosen = st.selectbox("Reader", range(len(items)), key=pick_key, format_func=lambda k: names[k],
                              label_visibility="collapsed")
    item = items[chosen]
    if item.error:
        st.error(f"Could not complete this reader: {item.error}")
    elif not item.plan or not item.result:
        if item.result:
            skipped_notes(item.result)
        st.warning("None of this reader's favorites matched. Try more widely known titles, or add a hint like \"Dune (film)\".")
        if item.result and item.result.unresolved:
            st.caption("Couldn't match: " + ", ".join(item.result.unresolved) + ". Try a specific title instead, for example \"Healer\".")
    else:
        plan, res = item.plan, item.result
        skipped_notes(res)
        match_warnings(res)
        if blend_problem(res):
            st.caption("Combined-taste matching was skipped: " + blend_problem(res))
        if res.matches:
            st.caption("Matched: " + " · ".join(res.matches) + ". Wrong one? Add a hint like \"Dune (film)\".")
        left, right = st.columns([1, 1])
        with left:
            guess = " This reason is a guess; the business can correct it." if plan.reason_guessed else ""
            html(f'<div class="panel"><div class="lbl">Likely reason</div><h3>{theme.esc(plan.slump_label)}</h3>'
                 f"<p>{theme.esc(res.way_back)}{theme.esc(guess)}</p></div>")
            html(theme.pick_card(res.picks[0], badge="Recommended", genre_of=res.genre_of))
            html(theme.alternates(plan.alternates))
        with right:
            html('<div class="panel"><div class="lbl">Ready-to-send message</div></div>')
            st.code(plan.message, language=None, wrap_lines=True)
            st.caption(f"Channel: {ss.get('batch_channel', channel)} · written by "
                       + ("Gemini" if plan.message_source == "gemini" else "a template"))
            html('<div class="panel"><div class="lbl">Evidence behind the pick</div><p>'
                 + "<br>".join("• " + theme.esc(line) for line in plan.evidence) + "</p></div>")
            svg = theme.taste_graph_svg(res.picks, res.resolved, res.genre_of)
            if svg:
                html('<div class="panel"><div class="lbl">Taste graph</div>' + svg + "</div>")
        with st.expander("How the agent got here (trace)"):
            html(theme.trace_html(res.trace))
    with st.expander(f"Edit {item.row.name}'s favorites and update this plan"):
        reader_editor(item.row, f"ed2_{id(items)}_{chosen}", replan)
    st.download_button("Download all plans (CSV)", business.export_csv(items, ss.get("batch_channel", channel)),
                       "win-back-plans.csv", "text/csv")
    st.caption("These are suggestions built on taste data. They make no claim about churn or conversion.")


# ========================================================================== reader


def run_reader() -> None:
    favs, reason_key, minutes = ss.r_inputs
    ss.r_result = agent.run(favs, reason_key, minutes or None, qloo, llm,
                            exclude_ids=ss.excl_ids, exclude_names=ss.excl_names)


def page_reader() -> None:
    top, right = st.columns([3, 2])
    with top:
        html(theme.hero("Find the book that ends the slump.",
                        "This is what your reader sees: a path from what they love to their next read."))
    with right:
        chips()
    demo_banner()

    html('<div class="note">Try a demo reader</div>')
    cols = st.columns(len(mock_data.DEMO_READERS) + 1)
    for col, (label, demo) in zip(cols, mock_data.DEMO_READERS.items()):
        if col.button(label.split(" (")[0], key=f"demo_{label}", width="stretch"):
            ss.r_inputs = (demo["favorites"], demo["reason"], demo["minutes"])
            ss.excl_ids, ss.excl_names, ss.worked = set(), set(), set()
            run_reader()
            st.rerun()

    cur = ss.r_inputs or ([], "taste_mismatch", 10)
    with st.form("reader_form"):
        favs = favorite_slots("rd_fav", list(cur[0]))
        keys = list(REASON_LABELS)
        reason = st.selectbox("What best describes your slump?", keys, format_func=lambda k: REASON_LABELS[k],
                              index=keys.index(cur[1]))
        st.markdown("**How much time can you read each day?** (optional)")
        minutes = time_picker("rd_time", int(cur[2] or 0))
        if st.form_submit_button("Break the slump", type="primary"):
            if not favs:
                st.error("Add at least one thing you love.")
            else:
                ss.r_inputs = (favs, reason, int(minutes))
                ss.excl_ids, ss.excl_names, ss.worked = set(), set(), set()
                with st.spinner("Asking the taste graph..."):
                    run_reader()
                st.rerun()

    result: AgentResult | None = ss.r_result
    if result is None:
        st.info("Pick a demo reader or fill in what you love, then press the button.")
        return
    if result.mode == "live-fallback-mock":
        st.error("Qloo could not be reached, so this run used the built-in sample data (it only knows a few titles). "
                 + (f"Reason: {fallback_reason(result)}" if fallback_reason(result) else ""))
    skipped_notes(result)
    match_warnings(result)
    if blend_problem(result):
        st.caption("Combined-taste matching was skipped: " + blend_problem(result))
    if result.matches:
        st.caption("We matched: " + " · ".join(result.matches)
                   + ". Wrong one? Add a hint, like \"Dune (film)\" or \"Taylor Swift (artist)\".")
    if result.unresolved:
        st.caption("Couldn't match: " + ", ".join(result.unresolved) + ". Try a specific title instead, for example \"Healer\".")
    if not result.picks:
        if qloo.is_mock:
            st.error("No picks found. Demo mode only knows a few titles (listed above). Add your Qloo key to try any title.")
        else:
            st.error("No picks found. Add more favorites or try widely known titles.")
        return

    html(f'<div class="panel"><div class="lbl">Diagnosis</div><h3>{theme.esc(result.slump_label)}</h3>'
         f"<p><b>Your way back in:</b> {theme.esc(result.way_back)}</p></div>")
    cols = st.columns(3)
    for n, (col, pick) in enumerate(zip(cols, result.picks)):
        c = pick.candidate
        with col:
            html(theme.pick_card(pick, badge="Recommended" if n == 0 else "Also consider", genre_of=result.genre_of))
            b1, b2 = st.columns(2)
            if b1.button("Worked 👍", key=f"w_{c.id}", width="stretch"):
                ss.worked.add(c.id)
            if b2.button("Not for me 👎", key=f"n_{c.id}", width="stretch"):
                ss.excl_ids.add(c.id)
                ss.excl_names.add(c.name)
                run_reader()
                st.rerun()
            if c.id in ss.worked:
                st.caption("Noted as a good pick.")
    if ss.excl_names:
        st.caption("Skipping for this session: " + ", ".join(sorted(ss.excl_names)))
    svg = theme.taste_graph_svg(result.picks, result.resolved, result.genre_of)
    if svg:
        html('<div class="panel"><div class="lbl">Taste-graph evidence</div><h3>How your favorites led here</h3>'
             + svg + '<p class="note">Solid line: that favorite\'s own list returned the book. Dashed line: the book fits your combined taste, which includes that favorite.</p></div>')
    with st.expander("How the agent got here (trace)", expanded=False):
        html(theme.trace_html(result.trace))


# ====================================================================== why qloo


def page_why() -> None:
    top, right = st.columns([3, 2])
    with top:
        html(theme.hero("Why Qloo?", "The same reader, asked two ways."))
    with right:
        chips()
    demo_banner()
    # Use the reader's own run when it found picks; otherwise a built-in example reader, so this page is never empty.
    own: AgentResult | None = ss.r_result
    if own is not None and own.picks and ss.r_inputs:
        result, favs, whose = own, list(ss.r_inputs[0]), "your reader view"
    else:
        label, demo = next(iter(mock_data.DEMO_READERS.items()))
        if ss.why_result is None or ss.why_result[0] != qloo.is_mock:
            res = agent.run(demo["favorites"], demo["reason"], demo["minutes"], qloo, llm)
            ss.why_result = (qloo.is_mock, res)
        result, favs, whose = ss.why_result[1], list(demo["favorites"]), "the example reader " + label.split(" (")[0]
    html(f'<div class="panel"><p>A plain AI chatbot knows what is popular, but it has no idea how <b>this</b> reader\'s '
         "favorite game, band or show connects to a book. Qloo's taste graph does, and every pick is checked "
         f"against what the taste graph returned.</p><p class='note'>Showing {theme.esc(whose)}: "
         f"{theme.esc(', '.join(favs))}.</p></div>")
    key = (tuple(favs), result.slump_label)
    if key not in ss.plain_cache:
        try:
            ss.plain_cache[key] = llm.plain_recs(favs, result.slump_label)
        except LLMError as exc:
            ss.plain_cache[key] = []
            st.error(f"The plain AI call failed: {exc}")
    plain = ss.plain_cache[key]
    a, b = st.columns(2)
    with a:
        html('<div class="panel"><div class="lbl">Plain AI, no taste data</div>'
             + "".join(f"<p>• <b>{theme.esc(p['title'])}</b> by {theme.esc(p['author'])}</p>" for p in plain)
             + '<p class="note">Popular books, with no link to this reader\'s favorites.</p></div>')
        if llm.mode == "mock":
            st.caption("No Gemini key: generic bestsellers stand in for a plain AI answer.")
    with b:
        if result.picks:
            html('<div class="panel"><div class="lbl">With Qloo</div>'
                 + "".join(f"<p>• <b>{theme.esc(p.candidate.name)}</b> by {theme.esc(p.candidate.author or 'unknown')}"
                           f'<br><span class="note">Linked to: {theme.esc(", ".join(spoken(p.candidate.seeds, result.genre_of)) or "your combined taste")}'
                           + (" · also fits all your favorites together" if p.candidate.blend and p.candidate.seeds else "")
                           + "</span></p>"
                           for p in result.picks)
                 + "</div>")
        else:
            html('<div class="panel"><div class="lbl">With Qloo</div><p>No picks came back for this reader.</p></div>')


# =========================================================================== shell


def shell() -> None:
    if ss.page not in PAGES:
        ss.page = "Business"
    if ss.last_page is None:
        ss.last_page = ss.page
    elif ss.last_page != ss.page:  # the person switched pages: remember where they came from
        ss.history.append(ss.last_page)
        ss.history[:] = ss.history[-10:]
        ss.last_page = ss.page

    with st.sidebar:
        html(theme.brand())
        if ss.user:
            st.caption(f"Signed in as **{ss.user.label}**  \n{ss.user.email}")
            st.button("Sign out", on_click=sign_out, width="stretch")
        else:
            st.caption("Guest session")
            if accounts.enabled and st.button("Sign in or create account", width="stretch"):
                ss.welcomed = False
                ss.onboarded = False
                st.rerun()
        st.button("Change view (business / reader)", on_click=change_view, width="stretch")
        st.caption("Picks come from Qloo's taste graph. The AI only chooses among them and writes the words.")

    nav_back, nav_pages, nav_view = st.columns([1, 5, 2])
    nav_back.button("← Back", on_click=go_back, disabled=not ss.history, width="stretch")
    nav_pages.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed",
                    format_func=lambda k: PAGE_LABELS[k])
    nav_view.button("Change view", on_click=change_view, width="stretch")
    {"Business": page_business, "Reader": page_reader, "Why Qloo?": page_why}[ss.page]()


if not ss.welcomed:
    welcome()
elif ss.onboarded:
    shell()
else:
    onboarding()
