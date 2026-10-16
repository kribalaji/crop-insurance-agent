"""End to end: synthetic ERA5 weather + sample policy PDFs through the supervisor."""

from datetime import date
from pathlib import Path

import pytest
from langgraph.types import Command

from crop_insurance_agent.config import Settings
from crop_insurance_agent.graph import build_graph
from crop_insurance_agent.graph.supervisor import next_step
from crop_insurance_agent.rag import PolicyStore
from crop_insurance_agent.schemas.advisory import AdvisoryDraft, AdvisoryNote
from crop_insurance_agent.schemas.policy import PolicyTerms
from tests.conftest import POINT, TARGET_YEAR, hash_embedder
from tests.factories import ABSOLUTE, HEAT, POLICY, RELATIVE, StubLLM

TERMS = PolicyTerms(
    crops=["Paddy (rice)"],
    risk_period_start=date(TARGET_YEAR, 6, 1),
    risk_period_end=date(TARGET_YEAR, 9, 30),
    sum_insured_per_ha=60000,
    triggers=[ABSOLUTE, RELATIVE, HEAT],
)
# Synthetic season: 146.4 mm vs 292.8 normal (50% deficit), GDU +13.3%.
# Absolute cover saturates (40%), relative pays (50-40)/(80-40) x 20 = 5%: 45% of
# Rs 1,20,000 = Rs 54,000.
ADVICE = AdvisoryDraft(
    headline="Your claim was approved for Rs 54,000.",
    explanation="Your rice got 146.4 mm of rain against the usual 292.8 mm.",
)


def _inputs(**weather_kw) -> dict:
    weather = dict(
        latitude=POINT[0],
        longitude=POINT[1],
        season_start=f"{TARGET_YEAR}-06-01",
        season_end=f"{TARGET_YEAR}-09-30",
        crop="rice",
    )
    return {
        "weather_request": weather | weather_kw,
        "policy_request": {"policy_id": POLICY},
        "claim_request": {"claim_id": "C-42", "area_ha": 2.0},
    }


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM({PolicyTerms: TERMS, AdvisoryDraft: ADVICE})


@pytest.fixture
def graph(data_dir: Path, policies_dir: Path, tmp_path: Path, llm: StubLLM):
    settings = Settings(
        _env_file=None, data_dir=data_dir, policies_dir=policies_dir, cache_dir=tmp_path / "c"
    )
    store = PolicyStore(settings.chroma_path, embedder=hash_embedder)
    return build_graph(settings=settings, llm=llm, store=store)


def test_full_claim_pauses_for_review_then_advises(graph, llm):
    config = {"configurable": {"thread_id": "claim-42"}}
    paused = graph.invoke(_inputs(), config)

    (pending,) = paused["__interrupt__"]
    assert pending.value["estimate"]["payout_amount"] == 54000.0
    assert paused["weather"].rainfall.deficit_pct == 50.0
    assert paused["policy"].triggered
    assert "advisory" not in paused
    assert graph.get_state(config).next == ("claim_agent",)

    done = graph.invoke(Command(resume={"approved": True, "reviewer": "officer-1"}), config)
    assert done["claim"].status == "approved" and done["claim"].payout_amount == 54000.0
    advisory = done["advisory"]
    assert isinstance(advisory, AdvisoryNote) and advisory.generated_by == "llm"
    assert advisory.headline == ADVICE.headline
    assert graph.get_state(config).next == ()
    assert [schema for schema, _ in llm.calls] == [PolicyTerms, AdvisoryDraft]


def test_weather_failure_is_explained_without_a_claim(graph, llm):
    llm.responses[AdvisoryDraft] = AdvisoryDraft(
        headline="We could not assess your claim.", explanation="No weather data was found."
    )
    done = graph.invoke(
        _inputs(latitude=20.0, longitude=85.0), {"configurable": {"thread_id": "x"}}
    )
    assert done["error"]["agent"] == "weather_agent"
    assert "outside the grid" in done["error"]["message"]
    assert "__interrupt__" not in done and "claim" not in done and "policy" not in done
    assert done["advisory"].facts["error"]["stage"] == "weather_agent"


def test_unknown_policy_is_explained(graph):
    inputs = _inputs() | {"policy_request": {"policy_id": "no_such_policy"}}
    done = graph.invoke(inputs, {"configurable": {"thread_id": "y"}})
    assert done["error"]["kind"] == "PolicyNotFoundError"
    assert "claim" not in done and done["advisory"] is not None


def test_routing_order():
    assert next_step({}) == "weather_agent"
    assert next_step({"weather": 1}) == "policy_agent"
    assert next_step({"weather": 1, "policy": 1}) == "claim_agent"
    assert next_step({"weather": 1, "policy": 1, "claim": 1}) == "advisory_agent"
    assert next_step({"weather": 1, "policy": 1, "claim": 1, "advisory": 1}) == "__end__"
    assert next_step({"error": {"agent": "weather_agent"}}) == "advisory_agent"
    assert next_step({"error": {"agent": "weather_agent"}, "advisory": 1}) == "__end__"
