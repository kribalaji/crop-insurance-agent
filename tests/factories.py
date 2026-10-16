"""Shared test objects matching the sample policy PDFs."""

from datetime import date

from crop_insurance_agent.schemas.policy import PolicyTrigger
from crop_insurance_agent.schemas.weather import (
    GDUAssessment,
    RainfallAssessment,
    WeatherAssessment,
)

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
