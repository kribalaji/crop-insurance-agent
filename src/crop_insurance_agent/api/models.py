"""Request and response models for the HTTP API."""

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from crop_insurance_agent.schemas.advisory import AdvisoryNote, Language
from crop_insurance_agent.schemas.claim import ClaimDecision, ClaimEstimate
from crop_insurance_agent.schemas.policy import PolicyAssessment
from crop_insurance_agent.schemas.weather import WeatherAssessment, WeatherRequest

# awaiting_approval: paused for a reviewer (POST /claims/{id}/decision)
# completed / failed: finished; failed means an agent error was explained to the farmer
# stalled: stopped by an unexpected error (e.g. LLM outage); fix it, then POST .../retry
ClaimStatus = Literal["awaiting_approval", "completed", "failed", "stalled"]


class ClaimSubmission(BaseModel):
    claim_id: str = Field(
        default_factory=lambda: uuid4().hex[:12], pattern=r"^[A-Za-z0-9_.-]{1,64}$"
    )
    weather: WeatherRequest
    policy_id: str
    area_ha: float = Field(gt=0)
    language: Language = "en"


class PendingReview(BaseModel):
    question: str
    assumptions: list[str]
    estimate: ClaimEstimate


class ClaimRunView(BaseModel):
    claim_id: str
    status: ClaimStatus
    pending_review: PendingReview | None = None
    weather: WeatherAssessment | None = None
    policy: PolicyAssessment | None = None
    claim: ClaimDecision | None = None
    advisory: AdvisoryNote | None = None
    error: dict[str, Any] | None = None


class ClaimSummary(BaseModel):
    claim_id: str
    status: ClaimStatus
    policy_id: str | None
    payout_amount: float | None
    error: str | None


class PolicyList(BaseModel):
    policies: list[str]
