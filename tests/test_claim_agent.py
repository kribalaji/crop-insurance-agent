import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from crop_insurance_agent.agents.claim_agent import claim_agent, estimate_claim
from crop_insurance_agent.graph.state import ClaimState
from crop_insurance_agent.schemas.claim import ClaimDecision, ClaimRequest
from crop_insurance_agent.schemas.policy import (
    PolicyAssessment,
    PolicyTrigger,
    TriggerEvaluation,
)
from tests.factories import ABSOLUTE, HEAT, POLICY, RELATIVE

REQUEST = ClaimRequest(claim_id="C-001", area_ha=2.0)


def _ev(trigger: PolicyTrigger, observed: float, met: bool, verified: bool = True):
    return TriggerEvaluation(trigger=trigger, observed=observed, met=met, quote_verified=verified)


def _policy(evaluations, triggered=True, sum_insured=60000.0) -> PolicyAssessment:
    return PolicyAssessment(
        policy_id=POLICY,
        crop="rice",
        crop_covered=True,
        sum_insured_per_ha=sum_insured,
        evaluations=evaluations,
        triggered=triggered,
        citations=[],
    )


# The sample season: 292.8 mm of rain, 50% deficit, GDU +13.3%.
SAMPLE = [_ev(ABSOLUTE, 292.8, True), _ev(RELATIVE, 50.0, True), _ev(HEAT, 13.3, False)]


def test_linear_payout_between_trigger_and_exit():
    est = estimate_claim(REQUEST, _policy(SAMPLE))

    absolute, relative = est.covers
    # (292.8 - 600) / (250 - 600) = 0.8777 of 40%
    assert absolute.method == "linear"
    assert absolute.payout_fraction == pytest.approx(0.8777, abs=1e-4)
    assert absolute.payout_pct == 35.11
    # (50 - 40) / (80 - 40) = 0.25 of 20%
    assert relative.payout_pct == 5.0
    assert est.total_payout_pct == 40.11
    assert est.sum_insured_total == 120000
    assert est.payout_amount == 48132.0
    assert not est.needs_manual_assessment and est.notes == []


def test_payout_saturates_beyond_exit_and_is_zero_at_trigger():
    beyond, at = estimate_claim(
        REQUEST, _policy([_ev(ABSOLUTE, 100.0, True), _ev(RELATIVE, 40.0, True)])
    ).covers
    assert (beyond.payout_fraction, beyond.payout_pct) == (1.0, 40.0)
    assert (at.payout_fraction, at.payout_pct) == (0.0, 0.0)


def test_cover_without_exit_pays_its_maximum():
    adversity = RELATIVE.model_copy(update={"threshold": 50, "exit_threshold": None})
    est = estimate_claim(REQUEST, _policy([_ev(adversity, 55.0, True)]))
    (cover,) = est.covers
    assert (cover.method, cover.payout_pct) == ("fixed", 20.0)
    assert any("no exit threshold" in n for n in est.notes)


def test_unverified_or_unmet_triggers_pay_nothing():
    est = estimate_claim(
        REQUEST, _policy([_ev(ABSOLUTE, 292.8, True, verified=False), _ev(HEAT, 13.3, False)])
    )
    assert est.covers == [] and est.payout_amount == 0


def test_untriggered_policy_pays_nothing():
    est = estimate_claim(REQUEST, _policy(SAMPLE, triggered=False))
    assert est.covers == [] and est.payout_amount == 0
    assert "not triggered" in est.notes[0]


def test_unpriceable_triggers_are_flagged():
    no_max = ABSOLUTE.model_copy(update={"max_payout_pct": None})
    wrong_side = RELATIVE.model_copy(update={"exit_threshold": 20})
    est = estimate_claim(REQUEST, _policy([_ev(no_max, 292.8, True), _ev(wrong_side, 50, True)]))
    assert est.covers == [] and est.needs_manual_assessment
    assert any("no maximum payout" in n for n in est.notes)
    assert any("wrong side" in n for n in est.notes)


def test_total_is_capped_at_sum_insured():
    big = ABSOLUTE.model_copy(update={"max_payout_pct": 80})
    est = estimate_claim(REQUEST, _policy([_ev(big, 100.0, True), _ev(big, 100.0, True)]))
    assert est.total_payout_pct == 100.0 and est.payout_amount == 120000
    assert any("capped" in n for n in est.notes)


def test_missing_sum_insured_needs_manual_assessment():
    est = estimate_claim(REQUEST, _policy(SAMPLE, sum_insured=None))
    assert est.total_payout_pct == 40.11
    assert est.payout_amount == 0 and est.needs_manual_assessment


@pytest.fixture
def graph():
    g = StateGraph(ClaimState)
    g.add_node("claim_agent", claim_agent)
    g.add_edge(START, "claim_agent")
    g.add_edge("claim_agent", END)
    return g.compile(checkpointer=InMemorySaver())


def _start(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "claim_request": REQUEST.model_dump(),
        "policy": _policy(SAMPLE).model_dump(mode="json"),
    }
    return config, graph.invoke(state, config)


def test_graph_pauses_for_approval_then_approves(graph):
    config, paused = _start(graph, "approve")

    (pending,) = paused["__interrupt__"]
    assert pending.value["estimate"]["payout_amount"] == 48132.0
    assert "reviewer" in pending.value["resume_schema"]["properties"]
    assert "claim" not in paused
    assert graph.get_state(config).next == ("claim_agent",)

    done = graph.invoke(
        Command(resume={"approved": True, "reviewer": "officer-1", "comment": "ok"}), config
    )
    claim = done["claim"]
    assert isinstance(claim, ClaimDecision)
    assert (claim.status, claim.payout_amount) == ("approved", 48132.0)
    assert claim.approval.reviewer == "officer-1"
    assert graph.get_state(config).next == ()


def test_rejection_pays_nothing(graph):
    config, _ = _start(graph, "reject")
    done = graph.invoke(Command(resume={"approved": False, "reviewer": "officer-2"}), config)
    assert (done["claim"].status, done["claim"].payout_amount) == ("rejected", 0.0)
    assert done["claim"].estimate.payout_amount == 48132.0  # kept for the audit trail


def test_malformed_approval_is_refused(graph):
    config, _ = _start(graph, "bad")
    with pytest.raises(ValidationError):
        graph.invoke(Command(resume={"approved": True}), config)  # no reviewer
    assert graph.get_state(config).next == ("claim_agent",)  # still awaiting a decision
