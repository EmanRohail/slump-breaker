from __future__ import annotations

import pytest

from slump_breaker import agent, business, mock_data
from slump_breaker.llm import MockLLM
from slump_breaker.qloo import QlooClient, QlooUnavailable


class FakeResponse:
    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

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
    result = agent.run(["Hades", "Totally Unknown Thing"], "burnout", None, mock_client(), MockLLM())
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

    result = agent.run(["Hades", "Hozier"], "burnout", None, mock_client(), HallucinatingLLM())
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
    assert sleeps == [1.0]
    client.resolve("Hades")  # served from cache
    assert len(session.calls) == 2


def test_persistent_429_falls_back_to_labeled_mock():
    session = FakeSession([FakeResponse(429)] * 10)
    client = QlooClient(api_key="k", session=session, max_retries=2, sleep=lambda s: None)
    result = agent.run(["Hades", "Hozier"], "burnout", None, client, MockLLM())
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
