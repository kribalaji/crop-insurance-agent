"""Shared test objects: sample-policy triggers, agent outputs, graph/API inputs and a stub LLM."""

from datetime import date

from crop_insurance_agent.schemas.advisory import AdvisoryDraft
from crop_insurance_agent.schemas.claim import (
    ApprovalDecision,
    ClaimDecision,
    ClaimEstimate,
    CoverPayout,
)
from crop_insurance_agent.schemas.policy import (
    PolicyAssessment,
    PolicyTerms,
    PolicyTrigger,
    TriggerEvaluation,
)
from crop_insurance_agent.schemas.weather import (
    GDUAssessment,
    RainfallAssessment,
    WeatherAssessment,
)
from tests.conftest import POINT, TARGET_YEAR

POLICY = "sample_rwbcis_paddy_thanjavur"

ABSOLUTE = PolicyTrigger(
    name="Deficit rainfall (absolute)",
    metric="rainfall_actual_mm",
    comparator="<",
    threshold=600,
    exit_threshold=250,
    max_payout_pct=40,
    source_quote="Deficit rainfall cover applies when cumulative rainfall during the risk "
    "period is less than 600 mm.",
    page=1,
)
RELATIVE = PolicyTrigger(
    name="Deficit rainfall (relative)",
    metric="rainfall_deficit_pct",
    comparator=">=",
    threshold=40,
    exit_threshold=80,
    max_payout_pct=20,
    source_quote="Additional deficit cover applies when the seasonal rainfall deficit "
    "relative to the long-period normal is 40 percent or more",
    page=2,
)
HEAT = PolicyTrigger(
    name="High temperature",
    metric="gdu_deviation_pct",
    comparator=">",
    threshold=15,
    exit_threshold=30,
    max_payout_pct=15,
    source_quote="High temperature cover applies when cumulative growing degree days over "
    "the risk period exceed the normal by more than 15 percent.",
    page=2,
)


def make_weather(**kw) -> WeatherAssessment:
    base = dict(
        source="era5",
        latitude=10.79,
        longitude=79.14,
        grid_latitude=10.75,
        grid_longitude=79.25,
        season_start=date(2023, 6, 1),
        season_end=date(2023, 9, 30),
        crop="rice",
        rainfall=RainfallAssessment(
            actual_mm=292.8,
            normal_mm=585.6,
            departure_pct=-50.0,
            deficit_pct=50.0,
            category="deficient",
        ),
        gdu=GDUAssessment(tbase_c=10, actual=2074, normal=1830, deviation=244, deviation_pct=13.3),
        baseline_years=[2018, 2019, 2020, 2021, 2022],
        data_coverage_pct=100.0,
    )
    return WeatherAssessment(**(base | kw))


def make_policy_assessment() -> PolicyAssessment:
    evs = [
        TriggerEvaluation(trigger=ABSOLUTE, observed=292.8, met=True, quote_verified=True),
        TriggerEvaluation(trigger=RELATIVE, observed=50.0, met=True, quote_verified=True),
        TriggerEvaluation(trigger=HEAT, observed=13.3, met=False, quote_verified=True),
    ]
    return PolicyAssessment(
        policy_id=POLICY,
        crop="rice",
        crop_covered=True,
        sum_insured_per_ha=60000,
        evaluations=evs,
        triggered=True,
        citations=[],
    )


def make_claim_decision(approved=True, assumptions=(), comment=None) -> ClaimDecision:
    cover = CoverPayout(
        name=ABSOLUTE.name,
        metric=ABSOLUTE.metric,
        observed=292.8,
        threshold=600,
        exit_threshold=250,
        max_payout_pct=40,
        method="linear",
        payout_fraction=0.8777,
        payout_pct=35.11,
    )
    est = ClaimEstimate(
        claim_id="C-1",
        policy_id=POLICY,
        crop="rice",
        area_ha=2,
        sum_insured_per_ha=60000,
        sum_insured_total=120000,
        covers=[cover],
        total_payout_pct=40.11,
        payout_amount=48132.0,
        needs_manual_assessment=False,
        assumptions=list(assumptions),
    )
    approval = ApprovalDecision(approved=approved, reviewer="officer-1", comment=comment)
    return ClaimDecision(
        estimate=est,
        approval=approval,
        status="approved" if approved else "rejected",
        payout_amount=48132.0 if approved else 0.0,
    )


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


def graph_inputs(**weather_kw) -> dict:
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


APPROVE = {"approved": True, "reviewer": "officer-1", "comment": "rain gauge agrees"}


def api_submission(claim_id="C-1", **weather_kw) -> dict:
    weather = {
        "latitude": POINT[0],
        "longitude": POINT[1],
        "season_start": f"{TARGET_YEAR}-06-01",
        "season_end": f"{TARGET_YEAR}-09-30",
        "crop": "rice",
    }
    return {
        "claim_id": claim_id,
        "weather": weather | weather_kw,
        "policy_id": POLICY,
        "area_ha": 2.0,
    }


class StubLLM:
    """Stands in for a chat model: answers per structured-output schema.

    Each response is either a fixed object or a callable taking the messages.
    Every call is recorded in `calls` as (schema, messages).
    """

    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.calls: list = []

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda

        def run(messages):
            self.calls.append((schema, messages))
            response = self.responses[schema]
            return response(messages) if callable(response) else response

        return RunnableLambda(run)
