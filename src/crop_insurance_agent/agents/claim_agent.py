"""claim_agent: price the triggered covers, then wait for a human decision.

Payouts follow the standard parametric rule, computed in code with no LLM:
each met and verified trigger pays max_payout_pct scaled linearly from 0 at the
trigger to 1 at the exit; a trigger without an exit pays its maximum. The total
is capped at 100% of the sum insured.

The LangGraph node computes the estimate and calls `interrupt()` with it. A
reviewer resumes the graph with `Command(resume=ApprovalDecision)`. On resume the
node runs again from the top, which is safe because the estimate is deterministic.
The node reads `claim_request` and `policy` from the graph state and writes `claim`.
"""

from collections.abc import Mapping
from typing import Any

from langgraph.types import interrupt

from crop_insurance_agent.schemas.claim import (
    MAX_TOTAL_PAYOUT_PCT,
    ApprovalDecision,
    ClaimDecision,
    ClaimEstimate,
    ClaimRequest,
    CoverPayout,
)
from crop_insurance_agent.schemas.policy import PolicyAssessment, TriggerEvaluation


def _exit_on_far_side(ev: TriggerEvaluation) -> bool:
    """For '<' covers the exit must lie below the trigger; for '>' covers above it."""
    t = ev.trigger
    assert t.exit_threshold is not None
    if t.comparator in ("<", "<="):
        return t.exit_threshold < t.threshold
    return t.exit_threshold > t.threshold


def price_cover(ev: TriggerEvaluation) -> CoverPayout | str:
    """Payout for one met, verified trigger, or the reason it cannot be priced."""
    t = ev.trigger
    assert ev.observed is not None
    if t.max_payout_pct is None:
        return f"{t.name!r} is met but the policy states no maximum payout."

    assumption = None
    if t.exit_threshold is None:
        method, fraction = "fixed", 1.0
        assumption = (
            f"DEFAULT ASSUMPTION, not from the policy: {t.name!r} states no cut-off, so its "
            f"full maximum of {t.max_payout_pct:g}% of the sum insured is paid. The policy "
            "may intend a lower amount (e.g. 'up to'); confirm before approving."
        )
    elif not _exit_on_far_side(ev):
        return (
            f"{t.name!r} has exit {t.exit_threshold:g} on the wrong side of "
            f"trigger {t.threshold:g} for comparator {t.comparator!r}."
        )
    else:
        method = "linear"
        raw = (ev.observed - t.threshold) / (t.exit_threshold - t.threshold)
        fraction = min(1.0, max(0.0, raw))

    return CoverPayout(
        name=t.name,
        metric=t.metric,
        observed=ev.observed,
        threshold=t.threshold,
        exit_threshold=t.exit_threshold,
        max_payout_pct=t.max_payout_pct,
        method=method,
        payout_fraction=round(fraction, 4),
        payout_pct=round(fraction * t.max_payout_pct, 2),
        assumption=assumption,
    )


def estimate_claim(request: ClaimRequest, policy: PolicyAssessment) -> ClaimEstimate:
    covers: list[CoverPayout] = []
    notes: list[str] = []
    unpriced = False

    if not policy.triggered:
        notes.append("Policy not triggered; no payout is due.")
    else:
        for ev in policy.evaluations:
            if not (ev.met and ev.quote_verified):
                continue
            priced = price_cover(ev)
            if isinstance(priced, str):
                notes.append(priced)
                unpriced = True
            else:
                covers.append(priced)

    raw_total = sum(c.payout_pct for c in covers)
    total = min(raw_total, MAX_TOTAL_PAYOUT_PCT)
    if raw_total > MAX_TOTAL_PAYOUT_PCT:
        notes.append(f"Covers add up to {raw_total:g}%; capped at {MAX_TOTAL_PAYOUT_PCT:g}%.")

    si_ha = policy.sum_insured_per_ha
    si_total = None if si_ha is None else round(si_ha * request.area_ha, 2)
    if si_total is None and covers:
        notes.append("Sum insured not found in the policy; payout amount cannot be computed.")
        unpriced = True

    return ClaimEstimate(
        claim_id=request.claim_id,
        policy_id=policy.policy_id,
        crop=policy.crop,
        area_ha=request.area_ha,
        sum_insured_per_ha=si_ha,
        sum_insured_total=si_total,
        covers=covers,
        total_payout_pct=round(total, 2),
        payout_amount=0.0 if si_total is None else round(total / 100 * si_total, 2),
        needs_manual_assessment=unpriced,
        assumptions=[c.assumption for c in covers if c.assumption],
        notes=notes,
    )


def decide(estimate: ClaimEstimate, approval: ApprovalDecision) -> ClaimDecision:
    return ClaimDecision(
        estimate=estimate,
        approval=approval,
        status="approved" if approval.approved else "rejected",
        payout_amount=estimate.payout_amount if approval.approved else 0.0,
    )


def claim_agent(state: Mapping[str, Any]) -> dict[str, Any]:
    """LangGraph node: `claim_request` + `policy` in, pauses for review, `claim` out."""
    request = ClaimRequest.model_validate(state["claim_request"])
    policy = PolicyAssessment.model_validate(state["policy"])
    estimate = estimate_claim(request, policy)

    question = "Approve this claim payout?"
    if estimate.assumptions:
        question += " The amount relies on default assumptions not stated in the policy; see"
        question += " 'assumptions'."
    resume = interrupt(
        {
            "question": question,
            "assumptions": estimate.assumptions,
            "resume_schema": ApprovalDecision.model_json_schema(),
            "estimate": estimate.model_dump(mode="json"),
        }
    )
    return {"claim": decide(estimate, ApprovalDecision.model_validate(resume))}
