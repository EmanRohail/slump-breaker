from __future__ import annotations

import pytest

from slump_breaker import agent, business, mock_data
from slump_breaker.llm import MockLLM
from slump_breaker.qloo import QlooClient, QlooUnavailable


class FakeResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        if text:
            self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses in order; records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def mock_client() -> QlooClient:
    return QlooClient(api_key="")


@pytest.mark.parametrize("name", list(mock_data.DEMO_READERS))
def test_demo_readers_get_three_grounded_picks(name):
    demo = mock_data.DEMO_READERS[name]
    result = agent.run(demo["favorites"], demo["reason"], demo["minutes"], mock_client(), MockLLM())
    assert len(result.picks) == 3
    assert result.mode == "mock"
    assert not result.unresolved
    for pick in result.picks:
        assert pick.candidate.seeds, "every pick must trace back to a favorite"
        assert pick.candidate.source == "mock"


def test_too_long_prefers_short_books():
    demo = mock_data.DEMO_READERS["Omar (books feel too long)"]
    result = agent.run(demo["favorites"], "too_long", 15, mock_client(), MockLLM())
    assert result.picks[0].candidate.pages <= 300


def test_unknown_favorite_is_reported_not_guessed():
    result = agent.run(["Minecraft", "Totally Unknown Thing"], "burnout", None, mock_client(), MockLLM())
    assert result.unresolved == ["Totally Unknown Thing"]
    assert result.picks


def test_nothing_resolved_returns_no_picks():
    result = agent.run(["zzz", "qqq"], "burnout", None, mock_client(), MockLLM())
    assert result.picks == []
    assert result.unresolved == ["zzz", "qqq"]


def test_exclusions_remove_a_book_and_refill():
    demo = mock_data.DEMO_READERS["Maya (last pick wasn't her taste)"]
    first = agent.run(demo["favorites"], demo["reason"], None, mock_client(), MockLLM())
    banned = first.picks[0].candidate
    second = agent.run(
        demo["favorites"], demo["reason"], None, mock_client(), MockLLM(),
        exclude_ids={banned.id}, exclude_names={banned.name},
    )
    assert banned.id not in [p.candidate.id for p in second.picks]
    assert len(second.picks) == 3


def test_grounding_check_drops_invented_books():
    class HallucinatingLLM(MockLLM):
        def select_picks(self, candidates, slump, favorites, minutes):
            return [{"id": "made-up", "why": "x"}] + super().select_picks(candidates, slump, favorites, minutes)

    result = agent.run(["Harry Potter", "Studio Ghibli"], "burnout", None, mock_client(), HallucinatingLLM())
    ids = {p.candidate.id for p in result.picks}
    assert "made-up" not in ids
    assert any(s.tool == "grounding_check" and not s.ok for s in result.trace)
    assert len(result.picks) == 3


def test_favorites_cleaned_deduped_and_capped():
    assert agent.clean_favorites(["a", "A", " ", "b", "c", "d", "e", "f"]) == ["a", "b", "c", "d", "e"]


def test_retry_on_429_then_success_and_cache():
    payload = {"results": [{"entity_id": "E1", "name": "Hades", "types": ["urn:entity:videogame"]}]}
    session = FakeSession([FakeResponse(429), FakeResponse(200, payload)])
    sleeps = []
    client = QlooClient(api_key="k", session=session, sleep=sleeps.append)
    ent = client.resolve("Hades")
    assert ent and ent.id == "E1"
    assert sleeps[0] == 1.0  # the 429 backoff comes first; throttle waits are tiny
    client.resolve("Hades")  # served from cache
    assert len(session.calls) == 2


def test_persistent_429_falls_back_to_labeled_mock():
    session = FakeSession([FakeResponse(429)] * 10)
    client = QlooClient(api_key="k", session=session, max_retries=2, sleep=lambda s: None)
    result = agent.run(["Harry Potter", "Studio Ghibli"], "burnout", None, client, MockLLM())
    assert result.mode == "live-fallback-mock"
    assert len(result.picks) == 3
    assert all(p.candidate.source == "mock" for p in result.picks)


def test_non_retryable_error_raises_immediately():
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(401)]), sleep=lambda s: None)
    with pytest.raises(QlooUnavailable):
        client.resolve("Hades")


def test_live_book_parsing_keeps_seed_linkage():
    search = {"results": [{"entity_id": "S1", "name": "Hades"}]}
    insights = {
        "results": {
            "entities": [
                {"entity_id": "B1", "name": "Circe", "tags": [{"name": "mythology"}], "query": {"affinity": 0.9},
                 "properties": {"author": "Madeline Miller"}},
            ]
        }
    }
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, search), FakeResponse(200, insights)]))
    seed = client.resolve("Hades")
    recs = client.book_recs([seed])
    assert recs[0].name == "Circe" and recs[0].seeds == ["Hades"] and recs[0].affinity == 0.9
    assert recs[0].author == "Madeline Miller" and recs[0].tags == ["mythology"]


def test_business_plan_has_message_and_honest_evidence():
    demo = mock_data.DEMO_READERS["Sana (burned out)"]
    result = agent.run(demo["favorites"], demo["reason"], demo["minutes"], mock_client(), MockLLM())
    plan = business.build_plan(result, "Sana", MockLLM())
    assert plan and "Sana" in plan.message
    assert any("mock fixture" in line for line in plan.evidence)


# ------------------------------------------------------------ tool-calling loop


class ScriptedGemini:
    """Fake Gemini HTTP: replays model turns that contain function calls."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append(json)
        parts = [{"functionCall": {"name": n, "args": a}} for n, a in self.turns.pop(0)]
        return FakeResponse(200, {"candidates": [{"content": {"role": "model", "parts": parts}}]})


def _gemini(turns):
    from slump_breaker.llm import GeminiLLM

    sess = ScriptedGemini(turns)
    return GeminiLLM("k", session=sess, sleep=lambda s: None, full_agent=True), sess


def test_model_driven_loop_calls_tools_then_finishes():
    llm, sess = _gemini([
        [("resolve_entity", {"name": "Harry Potter"}), ("resolve_entity", {"name": "Studio Ghibli"})],
        [("get_book_recs", {"entity_name": "Harry Potter"}), ("get_book_recs", {"entity_name": "Studio Ghibli"})],
        [("finish", {"picks": [{"id": "mock:howl's moving castle", "why": "Linked to Harry Potter and Studio Ghibli."}]})],
    ])
    result = agent.run(["Harry Potter", "Studio Ghibli"], "burnout", None, mock_client(), llm)
    assert result.llm_mode == "gemini"
    assert result.picks[0].candidate.name == "Howl's Moving Castle"
    assert result.picks[0].why == "Linked to Harry Potter and Studio Ghibli."
    assert len(result.picks) == 3  # topped up from the taste-graph list
    assert [s.tool for s in result.trace if s.tool in ("resolve_entity", "get_book_recs")].count("get_book_recs") == 2
    # the model saw the tool results on its next turn
    assert any("functionResponse" in p for req in sess.requests[1:] for c in req["contents"] for p in c["parts"])


def test_model_cannot_pick_a_book_the_taste_graph_never_returned():
    llm, _ = _gemini([
        [("resolve_entity", {"name": "Minecraft"})],
        [("get_book_recs", {"entity_name": "Minecraft"})],
        [("finish", {"picks": [{"id": "invented", "why": "x"}]})],
    ])
    result = agent.run(["Minecraft"], "burnout", None, mock_client(), llm)
    assert "invented" not in [p.candidate.id for p in result.picks]
    assert any(s.tool == "grounding_check" and not s.ok for s in result.trace)


def test_model_failure_falls_back_to_fixed_plan():
    from slump_breaker.llm import GeminiLLM

    class Down:
        def post(self, *a, **k):
            return FakeResponse(500)

    llm = GeminiLLM("k", session=Down(), sleep=lambda s: None)
    result = agent.run(["Harry Potter", "Studio Ghibli"], "burnout", None, mock_client(), llm)
    assert result.llm_mode == "mock"
    assert len(result.picks) == 3
    assert any(s.tool == "agent_model" and not s.ok for s in result.trace)


def test_tool_for_unresolved_entity_is_rejected_not_trusted():
    llm, _ = _gemini([
        [("get_book_recs", {"entity_name": "Minecraft"})],  # skipped resolve_entity
        [("finish", {"picks": []})],
    ])
    result = agent.run(["Minecraft"], "burnout", None, mock_client(), llm)
    assert result.picks == []
    assert any(s.tool == "get_book_recs" and not s.ok for s in result.trace)


def test_model_that_never_finishes_is_stopped():
    llm, _ = _gemini([[("resolve_entity", {"name": "Minecraft"})]] * 10)
    result = agent.run(["Minecraft"], "burnout", None, mock_client(), llm)
    assert any(s.tool == "finish" and not s.ok for s in result.trace)


# ------------------------------------------------------------- business batch


def test_csv_parses_names_favorites_and_reason():
    rows, problems = business.parse_csv(business.template_csv())
    assert not problems
    assert [r.name for r in rows] == ["Maya", "Omar", "Sana"]
    assert rows[0].favorites == ["Stranger Things", "Grand Theft Auto V", "Taylor Swift"]
    assert (rows[0].reason_key, rows[0].reason_guessed) == ("taste_mismatch", False)
    assert (rows[2].reason_key, rows[2].reason_guessed) == ("lost_habit", True)  # blank -> labeled guess


def test_csv_reports_bad_input_instead_of_crashing():
    assert business.parse_csv("")[1]
    assert business.parse_csv("a,b\n1,2")[1]  # wrong header
    rows, problems = business.parse_csv("name,favorites\nAna,\n,Minecraft\nBo,Minecraft")
    assert [r.name for r in rows] == ["Bo"] and len(problems) == 2


def test_csv_row_limit():
    text = "name,favorites\n" + "\n".join(f"R{i},Minecraft" for i in range(business.MAX_ROWS + 5))
    rows, problems = business.parse_csv(text)
    assert len(rows) == business.MAX_ROWS and any("first" in p for p in problems)


def test_batch_runs_every_reader_and_exports_safely():
    rows, _ = business.parse_csv("name,favorites,reason\n=EVIL(),Minecraft; Harry Potter,burnout\nNoMatch,zzz,")
    items = business.run_batch(rows, mock_client(), MockLLM(), channel="sms", business="Page Turner")
    assert len(items) == 2
    assert items[0].plan and items[0].plan.message.startswith("Page Turner:")
    assert items[1].plan is None and not items[1].error
    out = business.export_csv(items, "sms")
    assert "'=EVIL()" in out and "\n=EVIL()" not in out  # formula neutralized
    assert "No match found" in out


def test_message_fits_the_channel():
    demo = mock_data.DEMO_READERS["Sana (burned out)"]
    result = agent.run(demo["favorites"], demo["reason"], None, mock_client(), MockLLM())
    push = business.build_plan(result, "Sana", MockLLM(), channel="push").message
    email = business.build_plan(result, "Sana", MockLLM(), channel="email", business="Page Turner").message
    assert len(push) < 80 and email.endswith("- Page Turner")


# ------------------------------- real Qloo response shapes (from the Oct 2026 probe)

REAL_SEARCH_HADES = {"results": [
    {"name": "Hades", "entity_id": "BAND", "types": ["urn:entity:artist"], "popularity": 0.96,
     "tags": [{"name": "Rap"}]},
    {"name": "Hades", "entity_id": "GAME", "types": ["urn:entity:videogame"], "popularity": 0.94},
    {"name": "Hades II", "entity_id": "GAME2", "types": ["urn:entity:videogame"], "popularity": 0.9},
]}

REAL_INSIGHTS = {"success": True, "results": {"entities": [
    {"name": "Circe", "entity_id": "B1", "type": "urn:entity", "subtype": "urn:entity:book",
     "properties": {"page_count": 393, "publication_year": 2018,
                    "image": {"url": "https://images.qloo.com/i/B1-420x-auto.jpg"}},
     "popularity": 0.99, "query": {"affinity": 0.86}, "disambiguation": "2018, Madeline Miller",
     "external": {"goodreads": [{"id": "1", "user_rating": 4.23, "user_ratings_count": 1221258}]}},
    {"name": "Cold Granite / Dying Light / Broken Skin (Logan McRae #1-3)", "entity_id": "B2",
     "subtype": "urn:entity:book", "properties": {"page_count": 1120}, "query": {"affinity": 0.9},
     "disambiguation": "2009, Stuart MacBride"},
    {"name": "Piranesi", "entity_id": "B3", "subtype": "urn:entity:book",
     "properties": {"page_count": 272, "image": {"url": "https://evil.example/x.jpg"}},
     "query": {"affinity": 0.8}, "disambiguation": "2020, Susanna Clarke"},
]}}


def test_search_without_hint_prefers_the_game_over_a_metal_band_by_priority():
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, REAL_SEARCH_HADES)]))
    ent = client.resolve("Hades")
    assert (ent.id, ent.type) == ("GAME", "urn:entity:videogame")


def test_hint_overrides_the_default_choice():
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, REAL_SEARCH_HADES)]))
    assert client.resolve("Hades (band)").id == "BAND"
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, REAL_SEARCH_HADES)]))
    assert client.resolve("Hades (game)").id == "GAME"


def test_real_insights_shape_gives_pages_cover_author_rating_and_drops_box_sets():
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, REAL_INSIGHTS)]))
    from slump_breaker.models import Entity

    recs = client.recs_for_seed(Entity(id="GAME", name="Hades"))
    assert [c.name for c in recs] == ["Circe", "Piranesi"]  # omnibus dropped
    circe = recs[0]
    assert (circe.author, circe.pages, circe.year) == ("Madeline Miller", 393, 2018)
    assert circe.image_url.startswith("https://images.qloo.com/")
    assert (circe.rating, circe.rating_count) == (4.23, 1221258)
    assert recs[1].image_url == ""  # images from other hosts are never rendered


def test_matches_are_reported_so_a_wrong_match_is_visible():
    client = QlooClient(api_key="k", session=FakeSession([FakeResponse(200, REAL_SEARCH_HADES), FakeResponse(200, REAL_INSIGHTS)]))
    result = agent.run(["Hades"], "burnout", None, client, MockLLM())
    assert result.matches == ["Hades → Hades (videogame)"]
    assert result.picks[0].candidate.pages <= 300  # burnout prefers the short book (Piranesi)


def test_card_html_escapes_everything_and_shows_cover():
    from slump_breaker import theme
    from slump_breaker.models import Candidate, Pick

    c = Candidate(id="1", name="<script>x</script>", author="<b>A</b>", seeds=["<i>s</i>"],
                  image_url="https://images.qloo.com/i/1.jpg", rating=4.4, rating_count=287)
    html = theme.pick_card(Pick(c, "<img onerror=1>"))
    assert "<script>" not in html and "<b>A</b>" not in html and "<i>s</i>" not in html and "<img onerror" not in html
    assert 'src="https://images.qloo.com/i/1.jpg"' in html and "Goodreads 4.4" in html


def test_hint_uses_qloos_types_filter_and_falls_back_when_empty():
    only_games = {"results": [REAL_SEARCH_HADES["results"][1]]}
    session = FakeSession([FakeResponse(200, only_games)])
    client = QlooClient(api_key="k", session=session)
    assert client.resolve("Hades (game)").id == "GAME"
    assert session.calls[0][1]["types"] == "urn:entity:videogame"

    session = FakeSession([FakeResponse(200, {"results": []}), FakeResponse(200, REAL_SEARCH_HADES)])
    client = QlooClient(api_key="k", session=session)
    assert client.resolve("Hades (book)").id == "GAME"  # nothing of that type: fall back, don't fail
    assert "types" not in session.calls[1][1]


def test_requests_are_spaced_to_respect_the_five_per_second_limit():
    session = FakeSession([FakeResponse(200, {"results": []})] * 3)
    sleeps = []
    client = QlooClient(api_key="k", session=session, sleep=sleeps.append, clock=lambda: 100.0)
    for name in ["a", "b", "c"]:
        client.resolve(name)
    assert len(sleeps) == 2 and all(abs(x - 0.21) < 1e-9 for x in sleeps)


def test_audience_labels_are_not_shown_as_themes():
    from slump_breaker.qloo import _tag_names

    tags = [{"name": "Romance", "type": "urn:tag:genre:media"}, {"name": "Loyal", "type": "urn:tag:audience:qloo"}]
    assert _tag_names({"tags": tags}) == ["Romance"]


# ------------------------------------------------------------- matching variants


def test_short_forms_and_hyphens_expand_to_full_titles():
    from slump_breaker.qloo import query_variants

    assert query_variants("gta-5")[0] == "grand theft auto V"
    assert "gta 5" in query_variants("gta-5")
    assert query_variants("COD")[0] == "call of duty"


def test_mock_mode_resolves_gta_5_and_stranger_things():
    client = mock_client()
    assert client.resolve("gta-5").name == "Grand Theft Auto V"
    assert client.resolve("GTA V").name == "Grand Theft Auto V"
    assert client.resolve("stranger-things").name == "Stranger Things"


def test_live_resolve_retries_with_the_expanded_name_and_rejects_poor_matches():
    def search(q):
        if q.lower().startswith("grand theft auto"):
            return {"results": [{"entity_id": "G5", "name": "Grand Theft Auto V", "types": ["urn:entity:videogame"], "popularity": 0.9}]}
        return {"results": [{"entity_id": "X", "name": "Gta Cafe", "types": ["urn:entity:brand"]}]}

    class S:
        calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append(params["query"])
            return FakeResponse(200, search(params["query"]))

    client = QlooClient(api_key="k", session=S(), sleep=lambda s: None, min_interval=0)
    assert client.resolve("gta-5").id == "G5"
    # nothing close in the results: say so instead of guessing
    class Junk(S):
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"results": [{"entity_id": "Z", "name": "Completely Different", "types": ["urn:entity:brand"]}]})

    assert QlooClient(api_key="k", session=Junk(), sleep=lambda s: None, min_interval=0).resolve("gta-5") is None


def test_junk_editions_filtered_and_editions_merged():
    from slump_breaker.qloo import _OMNIBUS, merge_into

    for bad in ["The Official Heartstopper Colouring Book", "Heartstopper: Volume 1-4 Box Set", "Dune Study Guide"]:
        assert _OMNIBUS.search(bad), bad
    assert not _OMNIBUS.search("Heartstopper: Volume 1")
    from slump_breaker.models import Candidate

    pool = {}
    merge_into(pool, [Candidate(id="a", name="The Great Gatsby", seeds=["X"])])
    merge_into(pool, [Candidate(id="b", name="Great Gatsby", seeds=["Y"])])
    assert len(pool) == 1 and sorted(pool["a"].seeds) == ["X", "Y"]


# ------------------------------------------------------------------- accounts


class _AuthResp:
    def __init__(self, code, body):
        self.status_code, self._b, self.content = code, body, b"x"

    def json(self):
        return self._b


class _AuthSession:
    def __init__(self, *resps):
        self.resps, self.sent = list(resps), []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.sent.append((method, url, json))
        return self.resps.pop(0)


def test_auth_sign_in_and_profile_roundtrip():
    from slump_breaker.auth import AuthClient

    s = _AuthSession(
        _AuthResp(200, {"access_token": "t", "user": {"id": "1", "email": "a@b.co", "user_metadata": {"role": "business"}}}),
        _AuthResp(200, {"user_metadata": {"role": "reader", "bname": "X"}}),
    )
    c = AuthClient("https://x.supabase.co", "anon", session=s)
    u = c.sign_in(" A@B.co ", "pw")
    assert u.email == "a@b.co" and u.profile["role"] == "business" and s.sent[0][2]["email"] == "a@b.co"
    c.save_profile(u, {"role": "reader", "bname": "X", "favs": "never saved"})
    assert s.sent[1][2] == {"data": {"role": "reader", "bname": "X"}}  # only profile keys leave the app


def test_auth_errors_are_friendly_and_validated():
    from slump_breaker.auth import AuthClient, AuthError

    c = AuthClient("https://x.supabase.co", "anon", session=_AuthSession(_AuthResp(400, {"error_description": "Invalid login credentials"})))
    with pytest.raises(AuthError, match="Wrong email or password"):
        c.sign_in("a@b.co", "pw")
    with pytest.raises(AuthError):
        c.sign_up("not-an-email", "longenough1")
    with pytest.raises(AuthError, match="at least"):
        c.sign_up("a@b.co", "short")
    assert not AuthClient("", "").enabled and not AuthClient("http://evil.example", "k").enabled


def test_signup_needing_email_confirmation_returns_no_user():
    from slump_breaker.auth import AuthClient

    c = AuthClient("https://x.supabase.co", "anon", session=_AuthSession(_AuthResp(200, {"id": "1", "email": "a@b.co"})))
    user, note = c.sign_up("a@b.co", "longenough1")
    assert user is None and "confirmation" in note


def test_time_dropdown_ranges_and_estimates():
    assert business.estimate_requests(business.sample_rows()) == 3 * 9
    assert business.MAX_ROWS >= 100


# ------------------------------------------------------- combined taste + limits


def test_blended_query_sends_all_favorites_in_one_call_and_marks_blend():
    from slump_breaker.models import Entity

    insights = {"results": {"entities": [{"entity_id": "B1", "name": "Circe", "query": {"affinity": 0.9}}]}}

    class S:
        calls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append(params)
            return FakeResponse(200, insights)

    s = S()
    client = QlooClient(api_key="k", session=s, sleep=lambda x: None, min_interval=0)
    got = client.recs_for_group([Entity(id="E1", name="A"), Entity(id="E2", name="B"), Entity(id="E3", name="C")])
    assert s.calls[0]["signal.interests.entities"] == "E1,E2,E3"
    assert got[0].blend == ["A", "B", "C"] and got[0].seeds == []
    assert client.recs_for_group([Entity(id="E1", name="A")]) == []  # one favorite: nothing to blend


def test_agent_runs_blended_step_and_picks_carry_all_favorites():
    result = agent.run(["Stranger Things", "Grand Theft Auto V", "Taylor Swift"], "burnout", None, mock_client(), MockLLM())
    assert any(s.tool == "get_blended_recs" and s.ok for s in result.trace)
    assert set(result.resolved) == {"Stranger Things", "Grand Theft Auto V", "Taylor Swift"}
    assert len(result.picks[0].candidate.fits) == 3


def test_graph_draws_every_favorite_even_without_a_link():
    from slump_breaker import theme

    result = agent.run(["Minecraft", "Harry Potter"], "burnout", None, mock_client(), MockLLM())
    svg = theme.taste_graph_svg(result.picks, result.resolved + ["Unlinked Show"])
    assert "Minecraft" in svg and "Harry Potter" in svg and "Unlinked Show" in svg and "stroke-dasharray" in svg


def test_csv_warns_when_more_than_five_favorites():
    rows, problems = business.parse_csv("name,favorites\nAna,a;b;c;d;e;f;g")
    assert len(rows[0].favorites) == 5 and any("first 5" in p for p in problems)


def test_combined_query_rejected_by_qloo_does_not_kill_the_run():
    class R(FakeResponse):
        text = '{"message":"bad signal"}'

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            if "," in str(params.get("signal.interests.entities", "")):
                return R(400)
            if url.endswith("/search"):
                q = params["query"]
                return FakeResponse(200, {"results": [{"entity_id": "id-" + q, "name": q, "types": ["urn:entity:tv_show"]}]})
            return FakeResponse(200, {"results": {"entities": [{"entity_id": "B1", "name": "Circe", "query": {"affinity": 0.9}}]}})

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    result = agent.run(["Avatar", "BTS"], "burnout", None, client, MockLLM())
    assert result.mode == "live" and result.picks and result.picks[0].candidate.name == "Circe"
    step = next(s for s in result.trace if s.tool == "get_blended_recs")
    assert not step.ok and "bad signal" in step.detail


def test_refused_entity_is_replaced_by_next_best_match_not_a_total_failure():
    class R(FakeResponse):
        text = '{"errors":[{"message":"All of the provided signal.interests.entities were invalid."}]}'

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            if url.endswith("/search"):
                return FakeResponse(200, {"results": [
                    {"entity_id": "BAD", "name": "Avatar", "types": ["urn:entity:movie"], "popularity": 0.9},
                    {"entity_id": "OK", "name": "Avatar", "types": ["urn:entity:tv_show"], "popularity": 0.1}]})
            ids = str(params["signal.interests.entities"])
            if "BAD" in ids:
                return R(400)
            return FakeResponse(200, {"results": {"entities": [{"entity_id": "B1", "name": "Circe", "query": {"affinity": 0.9}}]}})

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    result = agent.run(["Avatar"], "burnout", None, client, MockLLM())
    assert result.mode == "live" and result.picks and "Avatar" not in result.unresolved


def test_all_favorites_refused_reports_them_and_keeps_live_mode():
    class R(FakeResponse):
        text = "invalid"

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            if url.endswith("/search"):
                return FakeResponse(200, {"results": [{"entity_id": "X1", "name": "Healer", "types": ["urn:entity:tv_show"]}]})
            return R(400)

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    result = agent.run(["Healer"], "burnout", None, client, MockLLM())
    assert result.mode == "live" and result.picks == [] and result.unresolved == ["Healer"]
    assert any(s.tool == "get_book_recs" and not s.ok for s in result.trace)


def test_albums_and_places_are_not_chosen_over_a_usable_match():
    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"results": [
                {"entity_id": "ALB", "name": "Cyberpunk", "types": ["urn:entity:album"], "popularity": 0.9},
                {"entity_id": "PLC", "name": "Cyberpunk", "types": ["urn:entity:place"], "popularity": 0.9},
                {"entity_id": "GAME", "name": "Cyberpunk 2077", "types": ["urn:entity:videogame"], "popularity": 0.8}]})

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    assert client.resolve("Cyberpunk").id == "GAME"


def test_popular_series_book_beats_obscure_namesakes():
    # what Qloo really returned for "Twisted Love" (Oct 2026)
    twisted = {"results": [
        {"entity_id": "FILM", "name": "Twisted Love", "types": ["urn:entity:movie"], "popularity": 0},
        {"entity_id": "PLC", "name": "Twisted Love", "types": ["urn:entity:place"], "popularity": 0.85},
        {"entity_id": "OLD", "name": "Twisted Love", "types": ["urn:entity:book"], "popularity": 0.68},
        {"entity_id": "HUANG", "name": "Twisted Love (Twisted, #1)", "types": ["urn:entity:book"], "popularity": 0.9986}]}

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, twisted)

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    assert client.resolve("Twisted Love").id == "HUANG"
    assert client.resolve("twisted love (book)").id == "HUANG"


@pytest.mark.parametrize("name, results, want", [
    ("Circe", [("ART", "Circe", "artist", 0.6582), ("NOVEL", "Circe", "book", 0.9998)], "NOVEL"),
    ("Dune", [("SHOW", "Dune", "tv_show", 0.9162), ("FILM", "Dune", "movie", 0.9996),
              ("BOOK", "Dune (Dune, #1)", "book", 0.9995)], "FILM"),
])
def test_far_more_popular_title_beats_the_type_order(name, results, want):
    payload = {"results": [{"entity_id": i, "name": n, "types": [f"urn:entity:{t}"], "popularity": p}
                           for i, n, t, p in results]}

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, payload)

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    assert client.resolve(name).id == want


def test_genre_words_are_not_titles():
    client = QlooClient(api_key="k", session=FakeSession([]), sleep=lambda x: None, min_interval=0)
    assert client.resolve("kdrama") is None and client.resolve("Japan") is None
    result = agent.run(["cricket"], "burnout", None, mock_client(), MockLLM())
    assert result.unresolved == ["cricket"] and "genre" in result.trace[1].detail


def test_genre_word_is_swapped_for_well_known_titles():
    result = agent.run(["kdrama", "Healer"], "burnout", None, mock_client(), MockLLM())
    note = next(t for t in result.trace if t.tool == "genre_anchor")
    assert "Crash Landing on You" in note.detail and "Squid Game" in note.detail
    assert "kdrama" not in result.unresolved
    from slump_breaker.qloo import genre_anchors
    assert genre_anchors("Goblin kdrama") is None and genre_anchors("Dune") is None


def test_gemini_model_not_found_falls_back_to_another_model():
    from slump_breaker.llm import GeminiLLM

    class S:
        urls = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.urls.append(url)
            if "gemini-flash-latest" in url:
                return FakeResponse(404)
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})

        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"models": [{"name": "models/gemini-9-flash", "supportedGenerationMethods": ["generateContent"]}]})

    s = S()
    llm = GeminiLLM("k", model="gemini-flash-latest", session=s, sleep=lambda x: None)
    assert llm._post({"contents": []})["candidates"]
    assert llm.model == "gemini-9-flash"


def test_trailing_kind_words_become_a_type_hint():
    from slump_breaker.qloo import parse_hint

    assert parse_hint("Goblin kdrama") == ("Goblin", "tv_show")
    assert parse_hint("Healer k-drama") == ("Healer", "tv_show")
    assert parse_hint("Death Note anime") == ("Death Note", "tv_show")
    assert parse_hint("Squid Game") == ("Squid Game", None)  # real titles stay whole
    assert parse_hint("kdrama") == ("kdrama", None)


def test_hobbies_and_sports_are_not_titles():
    from slump_breaker.qloo import is_generic

    for word in ["cricket", "volleyball", "baking", "dessert", "murder", "webtoon", "period drama"]:
        assert is_generic(word)
    assert not is_generic("Goblin kdrama")


def test_place_only_match_retries_for_a_real_title():
    place = {"results": [{"entity_id": "P1", "name": "Swan Lake Resort", "type": "urn:entity:place"}]}
    movie = {"results": [{"entity_id": "M1", "name": "Barbie of Swan Lake", "type": "urn:entity:movie"}]}
    empty = {"results": []}

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            kind = (params or {}).get("types", "")
            if kind == "urn:entity:movie":
                return FakeResponse(200, movie)
            return FakeResponse(200, place if not kind else empty)

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    e = client.resolve("babrie in swan lake")
    assert e is not None and e.name == "Barbie of Swan Lake"


def test_gemini_discovers_a_usable_model_when_all_names_404():
    from slump_breaker.llm import GeminiLLM

    class S:
        urls = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.urls.append(url)
            if "gemini-3-flash" in url:
                return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
            return FakeResponse(404)

        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"models": [
                {"name": "models/gemini-3-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding", "supportedGenerationMethods": ["embedContent"]},
            ]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    assert llm._post({"contents": []})["candidates"] and llm.model == "gemini-3-flash"


def test_historical_cdrama_is_a_genre_not_a_show_called_historical():
    from slump_breaker.qloo import genre_anchors

    assert genre_anchors("historical cdrama")[0].startswith("Nirvana in Fire")
    assert genre_anchors("Healer kdrama") is None


def test_gemini_overload_on_one_model_tries_the_next():
    from slump_breaker.llm import GeminiLLM

    class S:
        urls = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.urls.append(url)
            if "gemini-flash-latest" in url:
                return FakeResponse(503)
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResponse(200, {"models": [{"name": "models/gemini-3-flash-preview", "supportedGenerationMethods": ["generateContent"]}]})

    s = S()
    llm = GeminiLLM("k", model="gemini-flash-latest", session=s, sleep=lambda x: None)
    assert llm._post({"contents": []})["candidates"] and llm.model == "gemini-3-flash-preview"


def test_message_uses_the_readers_genre_not_the_stand_in_titles():
    result = agent.run(["kdrama", "Dune"], "burnout", None, mock_client(), MockLLM())
    result.genre_of = {"Squid Game": "kdrama"}
    assert business.spoken(["Squid Game", "Dune", "Squid Game"], result.genre_of) == ["kdrama", "Dune"]


def test_taste_graph_and_card_name_the_genre_not_the_stand_ins():
    from slump_breaker import theme
    from slump_breaker.models import Candidate, Pick

    cand = Candidate(id="b1", name="Book", author="A", tags=[], affinity=0.9,
                     seeds=["Crash Landing on You", "Squid Game"], blend=["Crash Landing on You", "Squid Game", "Dune"])
    g = {"Crash Landing on You": "kdrama", "Squid Game": "kdrama"}
    svg = theme.taste_graph_svg([Pick(candidate=cand, why="x")], ["Crash Landing on You", "Squid Game", "Dune"], g)
    assert "kdrama" in svg and "Squid" not in svg and "Crash" not in svg
    assert "kdrama, Dune" in theme.pick_card(Pick(candidate=cand, why="x"), genre_of=g)


def test_bundled_test_csv_parses_and_never_crashes():
    from pathlib import Path

    text = Path(__file__).resolve().parent.parent.joinpath("tools", "test_readers.csv").read_text(encoding="utf-8")
    rows, problems = business.parse_csv(text)
    assert len(rows) >= 25 and any("skipped" in p for p in problems)
    items = business.run_batch(rows, mock_client(), MockLLM(), fast=True)
    assert not any(i.error for i in items)


def test_uncertain_match_is_flagged():
    from slump_breaker.models import AgentResult  # noqa: F401

    class C(QlooClient):
        def resolve(self, name):
            from slump_breaker.models import Entity
            return Entity(id="x", name="Sombra", type="urn:entity:artist", source="mock")

    result = agent.run(["sombr"], "burnout", None, C(api_key=""), MockLLM())
    assert any("Sombra" in u for u in result.uncertain)


def test_gemini_default_uses_one_model_call_to_choose_picks():
    from slump_breaker.llm import GeminiLLM

    class S:
        def __init__(self):
            self.bodies = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.bodies.append(json)
            cands = json["contents"][0]["parts"][0]["text"]
            first_id = cands.split('"id": "')[1].split('"')[0]
            part = {"functionCall": {"name": "finish", "args": {"picks": [{"id": first_id, "why": "Because of Harry Potter."}]}}}
            return FakeResponse(200, {"candidates": [{"content": {"role": "model", "parts": [part]}}]})

    s = S()
    llm = GeminiLLM("k", session=s, sleep=lambda x: None)
    result = agent.run(["Harry Potter", "Studio Ghibli"], "burnout", None, mock_client(), llm)
    assert len(s.bodies) == 1 and result.llm_mode == "gemini"  # lookups ran in code, ONE model call chose the picks
    assert result.picks[0].why == "Because of Harry Potter." and len(result.picks) == 3
    assert s.bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_model_that_rejects_thinking_setting_is_retried_without_it():
    from slump_breaker.llm import GeminiLLM

    class R400(FakeResponse):
        text = "Invalid value at generation_config.thinking_config"

    class S:
        n = 0

        def post(self, url, headers=None, json=None, timeout=None):
            S.n += 1
            if "generationConfig" in json:
                return R400(400)
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    assert llm.write_message("A", "T", "B", ["x"], "way") == "hello" and llm._no_thinking


def test_batch_runs_readers_in_parallel_and_keeps_order():
    rows, _ = business.parse_csv(business.template_csv())
    seen = []
    items = business.run_batch(rows, mock_client(), MockLLM(), progress=lambda i, n: seen.append((i, n)))
    assert [i.row.name for i in items] == [r.name for r in rows]
    assert seen[-1] == (len(rows), len(rows)) and all(i.plan for i in items)


def test_gemini_never_waits_longer_than_its_time_budget():
    from slump_breaker.llm import GeminiLLM, LLMError
    import time as _t

    class Slow:
        def post(self, url, headers=None, json=None, timeout=None):
            raise __import__("requests").Timeout("hung")

        def get(self, *a, **k):
            raise __import__("requests").Timeout("hung")

    llm = GeminiLLM("k", session=Slow(), sleep=lambda x: None)
    start = _t.time()
    try:
        llm._post({"contents": []})
        assert False
    except LLMError as exc:
        assert "network error" in str(exc)
    assert _t.time() - start < 2  # a hung service fails at once, it does not stack up waits


def test_short_show_name_finds_the_full_title_not_a_lookalike():
    kdrama = {"results": [{"entity_id": "K1", "name": "Guardian: The Lonely and Great God", "type": "urn:entity:tv_show"}]}
    slayer = {"results": [{"entity_id": "S1", "name": "Goblin Slayer", "type": "urn:entity:tv_show"}]}

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            q = (params or {}).get("query", "").lower()
            return FakeResponse(200, kdrama if "guardian" in q or "lonely" in q else slayer)

    client = QlooClient(api_key="k", session=S(), sleep=lambda x: None, min_interval=0)
    e = client.resolve("Goblin kdrama")
    assert e is not None and e.name == "Guardian: The Lonely and Great God"
    result = agent.run(["Goblin kdrama"], "burnout", None, client, MockLLM())
    assert result.uncertain == []  # the full title IS what "Goblin" means, so no warning


def test_gemini_quota_error_fails_fast_with_a_clear_reason():
    from slump_breaker.llm import GeminiLLM, LLMError

    class S:
        n = 0

        def post(self, url, headers=None, json=None, timeout=None):
            S.n += 1
            return FakeResponse(429)

        def get(self, *a, **k):
            return FakeResponse(200, {"models": []})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: (_ for _ in ()).throw(AssertionError("no waiting")))
    try:
        llm._post({"contents": []})
        assert False
    except LLMError as exc:
        assert "quota" in str(exc) and S.n == 2  # each model was tried once, no waiting


def test_a_model_out_of_quota_falls_through_to_one_that_still_has_some():
    from slump_breaker.llm import GeminiLLM

    class S:
        def post(self, url, headers=None, json=None, timeout=None):
            if "flash-lite" in url:
                return FakeResponse(429)
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    assert llm._post({"contents": []})["candidates"] and llm.model == "gemini-flash-latest"


def test_same_request_is_answered_from_the_cache_without_a_new_model_call():
    from slump_breaker.llm import GeminiLLM

    class S:
        n = 0

        def post(self, url, headers=None, json=None, timeout=None):
            S.n += 1
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": " hi "}]}}]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    args = ("Maya", "Dune", "Frank Herbert", ["Dune"], "a short one")
    assert llm.write_message(*args) == "hi" and llm.write_message(*args) == "hi" and S.n == 1


def test_combined_taste_lookup_is_fetched_with_the_rest():
    calls = []

    class C(QlooClient):
        def recs_for_group(self, seeds, take=10):
            calls.append("group")
            return []

    c = C(api_key="k", session=FakeSession([]), sleep=lambda x: None, min_interval=0)
    from slump_breaker.agent import _Run, _prefetch
    from slump_breaker.models import Entity, ToolCall
    run = _Run(client=c, exclude_ids=set(), exclude_names=set())
    run.entities = {"a": Entity("1", "A", "urn:entity:movie", "qloo"), "b": Entity("2", "B", "urn:entity:movie", "qloo")}
    _prefetch(run, [ToolCall("get_blended_recs", {}), ToolCall("get_blended_recs", {})])
    assert calls == ["group", "group"]


def test_vague_400_drops_the_thinking_setting_and_retries():
    from slump_breaker.llm import GeminiLLM

    class S:
        bodies = []

        def post(self, url, headers=None, json=None, timeout=None):
            S.bodies.append(dict(json))
            if "thinkingConfig" in (json.get("generationConfig") or {}):
                return FakeResponse(400, text='{"error": {"message": "Request contains an invalid argument."}}')
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    assert llm._text("hi") == "ok" and llm._no_thinking


def test_a_400_on_one_model_moves_on_to_the_next():
    from slump_breaker.llm import GeminiLLM

    class S:
        def post(self, url, headers=None, json=None, timeout=None):
            if "flash-lite" in url:
                return FakeResponse(400, text="bad")
            return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    llm = GeminiLLM("k", session=S(), sleep=lambda x: None)
    assert llm._post({"contents": []})["candidates"] and llm.model == "gemini-flash-latest"


class RoutedSession:
    """Answers by endpoint; records every call."""

    def __init__(self, tags=None, search=None, insights=None):
        self.answers = {"/v2/tags": tags or {"results": {"tags": []}}, "/search": search or {"results": []},
                        "/v2/insights": insights or REAL_INSIGHTS}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(200, next(v for k, v in self.answers.items() if url.endswith(k)))


def live(session) -> QlooClient:
    return QlooClient(api_key="k", session=session, sleep=lambda x: None, min_interval=0)


def test_book_genres_are_recognised_but_titles_and_screen_genres_are_not():
    from slump_breaker.qloo import looks_like_genre

    for g in ["dark romance", "romcom", "cozy mystery", "Psychological thriller", "mafia thriller (genre)"]:
        assert looks_like_genre(g), g
    for t in ["Twisted Love", "kdrama", "anime (genre)", "Dune (book)", "The Silent Patient"]:
        assert not looks_like_genre(t), t


def test_reader_slang_maps_to_qloo_tags_and_books_come_from_the_tag_filter():
    s = RoutedSession()
    client = live(s)
    e = client.resolve("dark romance")
    assert (e.id, e.type) == ("urn:tag:genre:media:dark_romance", "urn:tag:genre")
    assert s.calls == []  # the map needs no request
    client.recs_for_seed(e)
    params = s.calls[-1][1]
    assert params["filter.tags"] == e.id and params["operator.filter.tags"] == "intersection"
    assert "signal.interests.entities" not in params


def test_other_genres_are_looked_up_in_qloos_tags():
    tags = {"results": {"tags": [{"name": "Mystery", "id": "urn:tag:genre:media:mystery"},
                                 {"name": "Cozy Mystery", "id": "urn:tag:genre:media:cozy_mystery"}]}}
    client = live(RoutedSession(tags=tags))
    assert client.resolve("cozy mystery").id == "urn:tag:genre:media:cozy_mystery"
    assert client.resolve("Cozy mystery (genre)").name == "Cozy Mystery"


def test_auto_detect_only_takes_a_near_exact_tag_otherwise_searches_the_title():
    tags = {"results": {"tags": [{"name": "Romance", "id": "urn:tag:genre:media:romance"}]}}
    search = {"results": [{"entity_id": "BK", "name": "The Last Romance", "types": ["urn:entity:book"]}]}
    s = RoutedSession(tags=tags, search=search)
    assert live(s).resolve("The Last Romance").id == "BK"
    # chosen as a genre, the closest tag is accepted instead
    assert live(RoutedSession(tags=tags)).resolve("viking romance (genre)").id == "urn:tag:genre:media:romance"


def test_title_plus_genre_blend_filters_the_titles_taste_by_the_genre():
    from slump_breaker.models import Entity

    s = RoutedSession()
    client = live(s)
    genre = client.resolve("romcom")
    client.recs_for_group([Entity(id="FW", name="Fourth Wing", type="urn:entity:book"), genre])
    params = s.calls[-1][1]
    assert params["signal.interests.entities"] == "FW"
    assert params["filter.tags"] == genre.id
    assert client.recs_for_group([genre, client.resolve("dark romance")]) == []  # two genres: nothing to blend


def test_agent_uses_genre_tags_live_and_stand_ins_offline():
    result = agent.run(["dark romance"], "burnout", None, live(RoutedSession()), MockLLM())
    note = next(t for t in result.trace if t.tool == "genre_tags")
    assert "Dark romance" in note.detail
    assert not any(t.tool == "genre_anchor" for t in result.trace)
    assert result.picks and not result.unresolved
    offline = agent.run(["romance"], "burnout", None, mock_client(), MockLLM())
    assert any(t.tool == "genre_anchor" for t in offline.trace)


def test_tropes_and_their_other_spellings_become_famous_example_books():
    from slump_breaker.qloo import trope_anchors

    e2l = trope_anchors("enemies to lovers")
    assert e2l and all(a.endswith("(book)") for a in e2l)
    for other in ["Enemies-to-lovers", "rivals to lovers", "enemies to lovers (trope)", "hate to love"]:
        assert trope_anchors(other) == e2l, other
    assert trope_anchors("Friends to Lovers") and trope_anchors("grumpy x sunshine") and trope_anchors("morally gray")
    assert trope_anchors("Twisted Love") is None and trope_anchors("enemies to lovers (movie)") is None


def test_agent_reports_a_trope_in_the_readers_words():
    result = agent.run(["enemies to lovers (trope)"], "burnout", None, mock_client(), MockLLM())
    note = next(t for t in result.trace if t.tool == "trope_anchor")
    assert note.detail.startswith("'enemies to lovers' is a trope") and "The Cruel Prince" in note.detail
    assert "enemies to lovers (trope)" not in result.unresolved


def test_an_unknown_trope_is_not_searched_as_a_title():
    s = RoutedSession()
    assert live(s).resolve("third act breakup (trope)") is None
    assert not any(url.endswith("/search") for url, _ in s.calls)


def test_a_trope_leaves_out_its_example_books_authors_unless_too_few_remain():
    search = {"results": [{"entity_id": "PP", "name": "Pride and Prejudice", "types": ["urn:entity:book"],
                           "disambiguation": "1813, Holly Black"}]}  # one example book; its author's sequels follow
    books = lambda *rows: {"results": {"entities": [  # noqa: E731
        {"entity_id": i, "name": n, "subtype": "urn:entity:book", "disambiguation": f"2020, {a}"} for i, n, a in rows]}}
    sequels_and_others = books(("Q", "The Queen of Nothing", "Holly Black"), ("A", "Book A", "Ann"),
                               ("B", "Book B", "Bea"), ("C", "Book C", "Cy"))
    result = agent.run(["slow burn (trope)"], "burnout", None,  # a trope without a usable book tag
                       live(RoutedSession(search=search, insights=sequels_and_others)), MockLLM())
    assert "The Queen of Nothing" not in [p.candidate.name for p in result.picks]
    assert any("example authors left out" in t.detail for t in result.trace)
    few = books(("Q", "The Queen of Nothing", "Holly Black"), ("A", "Book A", "Ann"))
    result = agent.run(["slow burn (trope)"], "burnout", None,
                       live(RoutedSession(search=search, insights=few)), MockLLM())
    assert "The Queen of Nothing" in [p.candidate.name for p in result.picks]


def test_tropes_with_a_book_tag_use_it_live_and_example_books_offline():
    s = RoutedSession()
    client = live(s)
    for typed in ["enemies to lovers (trope)", "Rivals to lovers", "enemies-to-lovers"]:
        assert client.resolve(typed).id == "urn:tag:genre:media:enemies_to_lovers", typed
    assert client.resolve("fake dating (trope)").id == "urn:tag:keyword:qloo:fake_dating"
    assert client.resolve("grumpy sunshine (trope)") is None  # no usable tag: the agent uses example books
    assert s.calls == []  # all from the map, no tag search
    result = agent.run(["enemies to lovers (trope)"], "burnout", None, live(RoutedSession()), MockLLM())
    note = next(t for t in result.trace if t.tool == "genre_tags")
    assert note.detail.startswith("'enemies to lovers' is a trope") and "Enemies to lovers" in note.detail
    offline = agent.run(["enemies to lovers (trope)"], "burnout", None, mock_client(), MockLLM())
    assert any(t.tool == "trope_anchor" for t in offline.trace)


def test_picks_spread_across_favorites_when_one_favorites_books_dominate():
    from slump_breaker.agent import _spread
    from slump_breaker.models import Candidate, Pick

    gta = [Candidate(id=f"g{i}", name=f"G{i}", seeds=["GTA"], blend=["GTA", "ST", "TS"]) for i in range(3)]
    st_ = Candidate(id="s", name="S", seeds=["ST"])
    ts = Candidate(id="t", name="T", seeds=["TS"])
    ranked = [*gta, st_, ts]
    picks, swapped = _spread([Pick(c, "why") for c in gta], ranked)
    assert [p.candidate.id for p in picks] == ["g0", "s", "t"] and swapped == 2
    assert picks[0].why == "why" and picks[1].why == ""
    # one favorite only: nothing to spread
    same, n = _spread([Pick(c, "why") for c in gta], gta)
    assert n == 0 and [p.candidate.id for p in same] == ["g0", "g1", "g2"]


def test_only_well_known_books_are_swapped_in():
    from slump_breaker.agent import _spread
    from slump_breaker.models import Candidate, Pick

    gta = [Candidate(id=f"g{i}", name=f"G{i}", seeds=["GTA"], rating_count=900_000) for i in range(3)]
    obscure = Candidate(id="t", name="Panama Papers", seeds=["TS"], rating_count=1_461)
    known = Candidate(id="s", name="Dead Poets Society", seeds=["ST"], rating_count=136_874)
    picks, swapped = _spread([Pick(c, "why") for c in gta], [*gta, obscure, known])
    assert [p.candidate.id for p in picks] == ["g0", "s", "g2"] and swapped == 1


def test_html_codes_in_qloo_names_are_decoded_once():
    data = {"results": {"entities": [{"entity_id": "R", "name": "Red, White &amp; Royal Blue", "subtype": "urn:entity:book",
                                      "disambiguation": "2019, Rosemary Valero-O&apos;Connell"}]}}
    from slump_breaker.models import Entity
    c = live(RoutedSession(insights=data)).recs_for_seed(Entity(id="x", name="x"))[0]
    assert (c.name, c.author) == ("Red, White & Royal Blue", "Rosemary Valero-O'Connell")


def test_little_known_picks_are_replaced_by_well_known_ones_from_the_same_results():
    from slump_breaker.agent import _lift
    from slump_breaker.models import Candidate, Pick

    sekiro = Candidate(id="k", name="Sekiro Side Story (serial)", seeds=["Minecraft"], rating_count=274)
    one_piece = Candidate(id="o", name="One Piece, Volume 22", seeds=["Dune"], rating_count=8_849)
    fine = Candidate(id="f", name="Circe", seeds=["Dune"], rating_count=12_000)  # above the line: kept
    hp_book = Candidate(id="h", name="Harry Potter", seeds=["Harry Potter"], rating_count=9_000_000)
    ghibli = Candidate(id="g", name="Howl's Moving Castle", seeds=["Studio Ghibli"], rating_count=200_000)
    offline = Candidate(id="m", name="Mock book", seeds=["X"])  # no ratings: never judged
    picks, lifted = _lift([Pick(sekiro, "w"), Pick(one_piece, "w"), Pick(offline, "w"), Pick(fine, "w")],
                          [sekiro, one_piece, offline, fine, hp_book, ghibli])
    # Sekiro -> Harry Potter (its favorite is not covered); One Piece -> the next well-known book; offline and
    # above-the-line picks stay
    assert [p.candidate.id for p in picks] == ["h", "g", "m", "f"]
    assert lifted == ["Sekiro Side Story", "One Piece, Volume 22"]
    assert picks[0].why == "" and picks[2].why == "w"
    alone, none = _lift([Pick(sekiro, "w")], [sekiro])  # nothing better: kept
    assert none == [] and alone[0].candidate.id == "k"
