"""Schemas for claim_agent: payout estimate, reviewer decision and final outcome."""

from typing import Literal

from pydantic import BaseModel, Field

from crop_insurance_agent.schemas.policy import TriggerMetric

# Total payout across all covers, as a percentage of the sum insured.
MAX_TOTAL_PAYOUT_PCT = 100.0


class ClaimRequest(BaseModel):
    claim_id: str = Field(min_length=1)
    area_ha: float = Field(gt=0, description="Insured area in hectares")


class CoverPayout(BaseModel):
    name: str
    metric: TriggerMetric
    observed: float
    threshold: float
    exit_threshold: float | None
    max_payout_pct: float
    method: Literal["linear", "fixed"] = Field(
        description="linear between trigger and exit, or the full maximum when no exit is set"
    )
    payout_fraction: float = Field(ge=0, le=1, description="Share of max_payout_pct paid")
    payout_pct: float = Field(description="Payout as a percentage of the sum insured")


class ClaimEstimate(BaseModel):
    claim_id: str
    policy_id: str
    crop: str
    area_ha: float
    sum_insured_per_ha: float | None
    sum_insured_total: float | None
    covers: list[CoverPayout]
    total_payout_pct: float = Field(description=f"Capped at {MAX_TOTAL_PAYOUT_PCT:g}%")
    payout_amount: float = Field(description="Rupees; 0 when the sum insured is unknown")
    needs_manual_assessment: bool = Field(
        description="A met trigger could not be priced, so the amount may be understated"
    )
    notes: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    """Resume value a human reviewer supplies to the paused graph."""

    approved: bool
    reviewer: str = Field(min_length=1)
    comment: str | None = None


class ClaimDecision(BaseModel):
    estimate: ClaimEstimate
    approval: ApprovalDecision
    status: Literal["approved", "rejected"]
    payout_amount: float = Field(description="Approved amount in rupees; 0 if rejected")
