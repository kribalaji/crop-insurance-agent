"""Schemas for policy_agent: what the LLM extracts and what the agent reports."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Each metric maps to one field of WeatherAssessment (see policy_agent.observed_value).
TriggerMetric = Literal["rainfall_actual_mm", "rainfall_deficit_pct", "gdu_deviation_pct"]
Comparator = Literal["<", "<=", ">", ">="]


class PolicyRequest(BaseModel):
    policy_id: str = Field(description="Policy document id: the PDF file name without .pdf")


class PolicyTrigger(BaseModel):
    name: str = Field(description="Short name of the cover, e.g. 'Deficit rainfall'")
    metric: TriggerMetric
    comparator: Comparator = Field(description="Direction in which the condition is met")
    threshold: float = Field(description="Trigger value of the metric")
    exit_threshold: float | None = Field(
        default=None, description="Metric value at which the maximum payout is reached"
    )
    max_payout_pct: float | None = Field(
        default=None, description="Maximum payout as a percentage of the sum insured"
    )
    source_quote: str = Field(description="The sentence stating the trigger, copied verbatim")
    page: int | None = Field(default=None, description="Page number of the source sentence")


class PolicyTerms(BaseModel):
    """Structured output requested from the LLM."""

    crops: list[str] = Field(default_factory=list, description="Crops the policy covers")
    risk_period_start: date | None = None
    risk_period_end: date | None = None
    sum_insured_per_ha: float | None = Field(
        default=None, description="Sum insured in rupees per hectare for the claimed crop"
    )
    triggers: list[PolicyTrigger] = Field(default_factory=list)


class Citation(BaseModel):
    policy_id: str
    page: int
    text: str


class TriggerEvaluation(BaseModel):
    trigger: PolicyTrigger
    observed: float | None = Field(description="Value of the metric from weather_agent")
    met: bool | None = Field(description="None when the observed value is undefined")
    quote_verified: bool = Field(description="source_quote appears in the retrieved text")


class PolicyAssessment(BaseModel):
    policy_id: str
    crop: str
    crop_covered: bool
    sum_insured_per_ha: float | None
    evaluations: list[TriggerEvaluation]
    triggered: bool = Field(
        description="Crop is covered and at least one verified trigger condition is met"
    )
    citations: list[Citation]
    notes: list[str] = Field(default_factory=list)
