"""Slump Breaker: Streamlit app. Run with: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from slump_breaker import agent, business, config, mock_data
from slump_breaker.llm import LLMError
from slump_breaker.models import AgentResult
from slump_breaker.qloo import QlooClient
from slump_breaker.slump_types import REASON_OPTIONS

st.set_page_config(page_title="Slump Breaker", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    .pick-card {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;}
    .pick-title {font-size: 1.15rem; font-weight: 650;}
    .muted {opacity: .7; font-size: .9rem;}
    .tag {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:999px;
          border:1px solid rgba(128,128,128,.4); font-size:.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_qloo() -> QlooClient:
    return QlooClient(api_key=config.qloo_api_key(), base_url=config.qloo_base_url())


@st.cache_resource
def get_llm():
    return agent.build_llm(config.gemini_api_key(), config.gemini_model())


qloo = get_qloo()
llm = get_llm()

# ----------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("excluded_ids", set())
ss.setdefault("excluded_names", set())
ss.setdefault("worked", set())
ss.setdefault("result", None)
ss.setdefault("inputs", None)

# ------------------------------------------------------------------------ header
st.title("📚 Slump Breaker")
st.caption("Get a reader out of a reading slump with picks grounded in Qloo's taste graph.")

if qloo.is_mock:
    st.warning(
        "**Mock mode:** no Qloo API key is configured, so recommendations come from a small built-in "
        "fixture, not real Qloo data. Add `QLOO_API_KEY` to go live.",
        icon="⚠️",
    )
if llm.mode == "mock":
    st.info("No Gemini key configured: explanations use a deterministic template instead of an LLM.")

# ------------------------------------------------------------------------ inputs
demo_names = ["(enter my own)"] + list(mock_data.DEMO_READERS.keys())
choice = st.selectbox("Try a demo reader", demo_names)
demo = mock_data.DEMO_READERS.get(choice)
if demo:
    st.caption(f"*{demo['story']}*")

reason_keys = [k for k, _ in REASON_OPTIONS]
reason_labels = {k: v for k, v in REASON_OPTIONS}
default_reason = reason_keys.index(demo["reason"]) if demo else 0

with st.form("reader_form"):
    left, right = st.columns([3, 2])
    with left:
        favs_text = st.text_area(
            "Things you love (up to 5, one per line: shows, games, music, films, books)",
            value="\n".join(demo["favorites"]) if demo else "",
            height=140,
            key=f"favs_{choice}",
        )
    with right:
        reason = st.selectbox(
            "What best describes your slump?",
            reason_keys,
            index=default_reason,
            format_func=lambda k: reason_labels[k],
            key=f"reason_{choice}",
        )
        minutes = st.number_input(
            "Minutes a day you could read (optional)",
            min_value=0,
            max_value=240,
            value=demo["minutes"] if demo else 0,
            step=5,
            key=f"minutes_{choice}",
        )
        reader_name = st.text_input("Reader's first name (for the business tab)", value="Alex")
    submitted = st.form_submit_button("Break the slump", type="primary")


def run_agent() -> None:
    favs, reason_key, mins, name = ss.inputs
    ss.result = agent.run(
        favorites=favs,
        reason_key=reason_key,
        minutes=mins or None,
        qloo=qloo,
        llm=llm,
        exclude_ids=ss.excluded_ids,
        exclude_names=ss.excluded_names,
    )


if submitted:
    favs = agent.clean_favorites(favs_text.splitlines())
    if not favs:
        st.error("Add at least one thing you love.")
    else:
        ss.inputs = (favs, reason, int(minutes), reader_name or "there")
        ss.excluded_ids, ss.excluded_names, ss.worked = set(), set(), set()
        with st.spinner("Asking the taste graph..."):
            run_agent()

result: AgentResult | None = ss.result
if result is None:
    st.stop()

# ------------------------------------------------------------------------- tabs
tab_reader, tab_business, tab_compare = st.tabs(["Reader view", "Business view", "With vs. without Qloo"])

if result.mode != "live":
    label = "mock fixture" if result.mode == "mock" else "mock fixture (Qloo was unavailable)"
    st.caption(f"Data source for this run: **{label}**.")

with tab_reader:
    if result.unresolved:
        st.warning("Couldn't match: " + ", ".join(result.unresolved) + ". Try a more widely known title.")
    if not result.picks:
        st.error("No picks found. Add more favorites or reset your feedback by running again.")
    else:
        st.subheader(f"Diagnosis: {result.slump_label}")
        st.success(f"**Your way back in:** {result.way_back}")
        for i, pick in enumerate(result.picks):
            c = pick.candidate
            tags = "".join(f'<span class="tag">{t}</span>' for t in c.tags[:5])
            seeds = ", ".join(c.seeds)
            meta = c.author or "Author not provided"
            if c.pages:
                meta += f" · {c.pages} pages"
            st.markdown(
                f"""<div class="pick-card"><div class="pick-title">{i + 1}. {c.name}</div>
                <div class="muted">{meta}</div><p>{pick.why}</p>{tags}
                <div class="muted">Linked to your favorites: {seeds}"""
                + (f" · affinity {c.affinity:.2f}" if c.affinity is not None else "")
                + "</div></div>",
                unsafe_allow_html=True,
            )
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Worked 👍", key=f"w_{c.id}"):
                ss.worked.add(c.id)
            if b2.button("Not for me 👎", key=f"n_{c.id}"):
                ss.excluded_ids.add(c.id)
                ss.excluded_names.add(c.name)
                run_agent()
                st.rerun()
            if c.id in ss.worked:
                st.caption("Noted as a good pick.")
        if ss.excluded_names:
            st.caption("Skipping for this session: " + ", ".join(sorted(ss.excluded_names)))

    with st.expander("How the agent got here (trace)"):
        for step in result.trace:
            st.write(("✅ " if step.ok else "⚠️ ") + f"**{step.tool}**: {step.detail}")

with tab_business:
    st.markdown(
        "For a reading app, book box or publisher: a lapsed reader becomes a **win-back plan**. "
        "This is a proposed layer on top of taste data; it makes no claims about churn or conversion."
    )
    try:
        plan = business.build_plan(result, ss.inputs[3], llm)
    except LLMError as exc:
        plan = None
        st.error(f"Could not write the message: {exc}")
    if plan is None:
        st.info("Run the agent to see a win-back plan.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Likely slump type", plan.slump_label)
        c2.metric("Suggested pick", plan.pick_title)
        st.markdown("**Ready-to-send re-engagement message**" + (" (template)" if plan.message_source == "mock" else ""))
        st.code(plan.message, language=None)
        st.markdown("**Evidence behind the pick**")
        for line in plan.evidence:
            st.write("• " + line)

with tab_compare:
    st.markdown("The same reader, asked two ways. Left: a plain LLM answer with no taste data. Right: Slump Breaker.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Plain LLM (no Qloo)")
        try:
            plain = llm.plain_recs(ss.inputs[0], result.slump_label)
        except LLMError as exc:
            plain = []
            st.error(f"Plain LLM call failed: {exc}")
        if llm.mode == "mock":
            st.caption("Mock mode: generic bestseller-style picks stand in for an LLM answer.")
        for p in plain:
            st.write(f"• **{p['title']}** by {p['author']}")
        st.caption("No link to this reader's favorites is available for these picks.")
    with col_b:
        st.markdown("#### Slump Breaker (with Qloo)")
        for pick in result.picks:
            c = pick.candidate
            st.write(f"• **{c.name}** by {c.author or 'unknown'}")
            st.caption(f"Linked to: {', '.join(c.seeds)}")
