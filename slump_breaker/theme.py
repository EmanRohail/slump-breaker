"""Look and feel: dark purple theme, plus small HTML builders.

Everything that can contain user or Qloo text goes through html.escape, because
these strings are rendered with unsafe_allow_html.
"""
from __future__ import annotations

from html import escape as esc

from .models import Pick, TraceStep, spoken

CSS = """
<style>
[data-testid="stAppViewContainer"]{background:radial-gradient(900px 500px at 85% -5%,#3a2275 0,transparent 60%),radial-gradient(700px 500px at 0% 100%,#2a1658 0,transparent 60%),#120c24;}
[data-testid="stHeader"]{background:transparent;}
/* Streamlit caps dropdown lists at 300px, which hides the last of the eight favorite types: show them all. */
[data-testid="stSelectboxVirtualDropdown"]{max-height:none!important;}
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"]{max-height:420px!important;}
#MainMenu,footer,[data-testid="stAppDeployButton"],[data-testid="stMainMenu"]{display:none!important;}
[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{visibility:visible!important;display:flex!important;opacity:1!important;color:#f1eaff!important;}
[data-testid="stExpandSidebarButton"] *,[data-testid="stSidebarCollapsedControl"] *{color:#f1eaff!important;}
[data-testid="stSidebar"]{background:rgba(18,12,36,.85);border-right:1px solid rgba(190,160,255,.18);}
.block-container{max-width:1180px;padding-top:2rem;}
h1,h2,h3{font-family:Georgia,'Times New Roman',serif!important;font-weight:400!important;letter-spacing:-.3px;}
.brand{font-family:Georgia,serif;font-size:1.5rem;display:flex;gap:.6rem;align-items:center;margin-bottom:1rem;}
.logo{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#b58cff,#7c5cff);display:grid;place-items:center;}
.hero{font-family:Georgia,serif;font-size:2.6rem;line-height:1.1;margin:0 0 .3rem 0;}
.sub{color:#a99bd0;font-size:1.05rem;margin-bottom:1rem;}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;}
.chip{padding:.35rem .9rem;border-radius:999px;border:1px solid rgba(190,160,255,.25);background:rgba(36,25,69,.8);font-size:.82rem;display:inline-flex;gap:.5rem;align-items:center;}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.dot.live{background:#7be0b0;}.dot.mock{background:#f3c969;}
.panel{background:linear-gradient(180deg,rgba(36,25,69,.85),rgba(28,19,54,.85));border:1px solid rgba(190,160,255,.18);border-radius:18px;padding:1rem 1.2rem;margin-bottom:1rem;}
.lbl{font-size:.72rem;letter-spacing:1.4px;color:#b58cff;text-transform:uppercase;margin-bottom:.3rem;}
.panel h3{margin:.1rem 0 .4rem 0;font-size:1.35rem;}
.panel p{color:#d9ccf5;line-height:1.5;margin:.3rem 0 0 0;}
.pick{border:1px solid rgba(190,160,255,.2);border-radius:14px;padding:.8rem;background:rgba(18,12,36,.55);min-height:400px;margin-bottom:.4rem;}
.cover{height:150px;border-radius:10px;display:flex;align-items:flex-end;padding:.6rem .7rem;font-family:Georgia,serif;font-size:1.1rem;line-height:1.15;color:#fff;}
.by{font-size:.85rem;color:#a99bd0;margin:.5rem 0 .3rem 0;}
.aff{display:inline-block;font-size:.75rem;padding:.1rem .6rem;border-radius:999px;background:rgba(124,92,255,.25);border:1px solid rgba(190,160,255,.25);margin-bottom:.4rem;}
.why{font-size:.88rem;color:#e6dcff;line-height:1.4;margin:.3rem 0;}
.tag{display:inline-block;font-size:.72rem;padding:.05rem .6rem;margin:.1rem .25rem .1rem 0;border-radius:999px;border:1px solid rgba(190,160,255,.25);color:#a99bd0;}
.badge{display:inline-block;font-size:.72rem;letter-spacing:1px;text-transform:uppercase;font-weight:600;padding:.2rem .7rem;border-radius:999px;margin-bottom:.5rem;background:linear-gradient(135deg,#b58cff,#7c5cff);color:#fff;}
.badge.soft{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.25);color:#fff;}
.pick.rec{border-color:#b58cff;box-shadow:0 0 0 1px rgba(181,140,255,.55),0 8px 30px rgba(124,92,255,.25);}
.alts{margin-top:.6rem;}.alts .alt{color:#fff;font-size:.92rem;padding:.35rem .7rem;margin:.25rem 0;border-radius:10px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.18);}
.alts .alt span{color:#fff;opacity:.75;}
.link{font-size:.76rem;color:#a99bd0;margin-top:.3rem;}
.panel svg{max-width:640px;display:block;margin:0 auto;}
.trace{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;}
.tr{flex:1 1 150px;border:1px solid rgba(190,160,255,.2);border-radius:12px;padding:.5rem .7rem;background:rgba(36,25,69,.6);font-size:.8rem;color:#a99bd0;}
.tr b{display:block;color:#f1eaff;font-size:.85rem;}
.tr.bad{border-color:rgba(243,201,105,.5);}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.cmp>div{border:1px solid rgba(190,160,255,.2);border-radius:12px;padding:.7rem .9rem;font-size:.9rem;}
.cmp .good{background:rgba(123,224,176,.08);border-color:rgba(123,224,176,.35);}
.step{font-size:.8rem;color:#a99bd0;letter-spacing:.5px;}
.note{font-size:.82rem;color:#a99bd0;}
[data-testid="InputInstructions"]{display:none!important;}
div.stButton>button,div.stDownloadButton>button{border-radius:12px;border:1px solid rgba(190,160,255,.3);}
@media (max-width:700px){.hero{font-size:1.9rem}.cmp{grid-template-columns:1fr}}
</style>
"""

_COVERS = [
    ("#7c5cff", "#d96cae"), ("#2f7bd8", "#7c5cff"), ("#c9569a", "#f0a15a"),
    ("#3aa17e", "#2f7bd8"), ("#8e5cff", "#40b6c9"), ("#d96c6c", "#7c5cff"),
]


def _cover(title: str) -> str:
    a, b = _COVERS[sum(ord(ch) for ch in title) % len(_COVERS)]
    return f"linear-gradient(135deg,{a},{b})"


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def hero(title: str, sub: str) -> str:
    return f'<div class="hero">{esc(title)}</div><div class="sub">{esc(sub)}</div>'


def brand() -> str:
    return '<div class="brand"><div class="logo">📖</div>Slump Breaker</div>'


def status_chips(qloo_mock: bool, llm_mode: str) -> str:
    q = ("mock", "Demo mode · sample data") if qloo_mock else ("live", "Live · Qloo taste graph")
    extra = '<span class="chip">✦ AI-written messages</span>' if llm_mode == "gemini" else ""
    return f'<div class="chips"><span class="chip"><span class="dot {q[0]}"></span>{q[1]}</span>{extra}</div>'


def alternates(items: list[tuple[str, str]]) -> str:
    """Other options, in white so they read clearly against the dark panel."""
    if not items:
        return ""
    rows = "".join(f'<div class="alt">{esc(t)}' + (f' <span>by {esc(a)}</span>' if a else "") + "</div>" for t, a in items)
    return f'<div class="alts"><div class="lbl">Other options</div>{rows}</div>'


def pick_card(pick: Pick, badge: str = "", genre_of: dict | None = None) -> str:
    c = pick.candidate
    aff = f'<span class="aff">Affinity {c.affinity:.2f}</span>' if c.affinity is not None else ""
    tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in c.tags[:4])
    meta = esc(c.author or "Author not provided")
    if c.year:
        meta += f" · {c.year}"
    if c.pages:
        meta += f" · {c.pages} pages"
    if c.image_url:  # cover served by Qloo; the URL is checked in qloo.py
        cover = (f'<div class="cover" style="background:{_cover(c.name)};padding:0;overflow:hidden">'
                 f'<img src="{esc(c.image_url)}" alt="Cover of {esc(c.name)}" loading="lazy" '
                 'style="width:100%;height:100%;object-fit:contain;background:rgba(0,0,0,.25)"></div>')
    else:
        cover = f'<div class="cover" style="background:{_cover(c.name)}">{esc(c.name)}</div>'
    rating = ""
    if c.rating is not None:
        count = f" · {c.rating_count:,} ratings" if c.rating_count else ""
        rating = f'<div class="link">Goodreads {c.rating:.1f}{count} (via Qloo)</div>'
    tag = f'<div class="badge{"" if badge == "Recommended" else " soft"}">{"★ " if badge == "Recommended" else ""}{esc(badge)}</div>' if badge else ""
    return (
        f'<div class="pick{" rec" if badge == "Recommended" else ""}">{tag}{cover}<div class="by"><b style="color:#f1eaff">{esc(c.name)}</b><br>{meta}</div>'
        f'{aff}<div class="why">{esc(pick.why)}</div>{tags}{rating}'
        f'<div class="link">Linked to: {esc(", ".join(spoken(c.fits, genre_of or {})))}</div></div>'
    )


def taste_graph_svg(picks: list[Pick], favorites: list[str] | None = None, genre_of: dict | None = None) -> str:
    """Every favorite on the left, each pick on the right.

    Solid line: the book came back from that favorite's own list. Dashed line:
    it came back for the combined taste that includes that favorite.
    """
    g = genre_of or {}  # "kdrama" is drawn once, not as the two stand-in titles behind it
    seeds: list[str] = spoken(favorites or [], g)  # every favorite is drawn, linked or not
    for p in picks:
        for s in spoken(p.candidate.fits, g):
            if s not in seeds:
                seeds.append(s)
    seeds = seeds[:5]
    if not seeds or not picks:
        return ""
    height = 40 + 56 * max(len(seeds), len(picks))

    def ys(n: int) -> list[float]:
        gap = (height - 40) / max(n, 1)
        return [20 + gap * i + gap / 2 for i in range(n)]

    sy, by = ys(len(seeds)), ys(len(picks))
    paths, nodes = [], []
    for j, p in enumerate(picks):
        for s in spoken(p.candidate.fits, g):
            if s in seeds:
                i = seeds.index(s)
                direct = s in spoken(p.candidate.seeds, g)
                dash = "" if direct else ' stroke-dasharray="5 5"'
                paths.append(
                    f'<path d="M150 {sy[i]:.0f} C265 {sy[i]:.0f} 265 {by[j]:.0f} 380 {by[j]:.0f}" '
                    f'stroke="#b58cff" stroke-opacity="{.75 if direct else .4}" stroke-width="1.6" fill="none"{dash}/>'
                )
    for i, s in enumerate(seeds):
        nodes.append(
            f'<rect x="20" y="{sy[i]-19:.0f}" width="130" height="38" rx="19" fill="#2c1f55" stroke="#b58cff" stroke-opacity=".5"/>'
            f'<text x="85" y="{sy[i]+5:.0f}" text-anchor="middle" font-size="13" fill="#f1eaff">{esc(_short(s, 16))}</text>'
        )
    for j, p in enumerate(picks):
        nodes.append(
            f'<rect x="380" y="{by[j]-19:.0f}" width="170" height="38" rx="19" fill="#3b2a78"/>'
            f'<text x="465" y="{by[j]+5:.0f}" text-anchor="middle" font-size="13" fill="#f1eaff">{esc(_short(p.candidate.name, 20))}</text>'
        )
    return (
        f'<svg viewBox="0 0 570 {height}" width="100%" role="img" aria-label="Favorites linked to recommended books">'
        + "".join(paths) + "".join(nodes) + "</svg>"
    )


def trace_html(steps: list[TraceStep]) -> str:
    cells = []
    for s in steps:
        cls = "tr" if s.ok else "tr bad"
        cells.append(f'<div class="{cls}"><b>{"✓" if s.ok else "⚠"} {esc(s.tool)}</b>{esc(s.detail)}</div>')
    return '<div class="trace">' + "".join(cells) + "</div>"
