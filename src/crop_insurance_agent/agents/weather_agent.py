"""weather_agent: seasonal rainfall deficit and GDU deviation for a farm location.

Deterministic by design: no LLM is involved, so the numbers feeding a claim are
reproducible and auditable. The LangGraph node reads `weather_request` from the
graph state and writes `weather`.
"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from crop_insurance_agent.config import Settings, get_settings
from crop_insurance_agent.schemas.weather import (
    GDUAssessment,
    RainfallAssessment,
    WeatherAssessment,
    WeatherRequest,
)
from crop_insurance_agent.weather.cache import get_point_daily
from crop_insurance_agent.weather.metrics import (
    daily_gdd,
    departure_pct,
    imd_rainfall_category,
    rainfall_deficit_pct,
    shifted_window,
)
from crop_insurance_agent.weather.sources import WeatherDataError

WMO_NORMAL_YEARS = 30


def _season_totals(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tbase_c: float
) -> tuple[float, float, float]:
    """(rain total, GDD total, coverage) over [start, end]; coverage is in 0..1."""
    days = pd.date_range(start, end, freq="D")
    window = df.reindex(days)
    complete = window[["precip_mm", "tmax_c", "tmin_c"]].notna().all(axis=1)
    coverage = float(complete.mean())
    rain = float(window["precip_mm"].sum())
    gdd = float(daily_gdd(window["tmax_c"], window["tmin_c"], tbase_c).sum())
    return rain, gdd, coverage


def assess_weather(request: WeatherRequest, settings: Settings | None = None) -> WeatherAssessment:
    settings = settings or get_settings()
    tbase = request.tbase_c
    assert tbase is not None  # filled by WeatherRequest's validator
    df = get_point_daily(
        settings.data_dir, settings.cache_dir, request.source, request.latitude, request.longitude
    )

    start, end = pd.Timestamp(request.season_start), pd.Timestamp(request.season_end)
    rain, gdd, coverage = _season_totals(df, start, end, tbase)
    if coverage < request.min_coverage:
        raise WeatherDataError(
            f"season {request.season_start}..{request.season_end} has only "
            f"{coverage:.0%} complete days (need {request.min_coverage:.0%})"
        )

    target_year = request.season_start.year
    first = request.baseline_start_year or df.index.min().year
    last = request.baseline_end_year or df.index.max().year
    baseline_years: list[int] = []
    base_rain: list[float] = []
    base_gdd: list[float] = []
    for year in range(first, last + 1):
        if year == target_year:
            continue
        b_start, b_end = shifted_window(
            request.season_start, request.season_end, year - target_year
        )
        r, g, cov = _season_totals(df, b_start, b_end, tbase)
        if cov >= request.min_coverage:
            baseline_years.append(year)
            base_rain.append(r)
            base_gdd.append(g)

    if len(baseline_years) < request.min_baseline_years:
        raise WeatherDataError(
            f"only {len(baseline_years)} usable baseline seasons between {first} and {last} "
            f"(need {request.min_baseline_years})"
        )

    normal_rain = sum(base_rain) / len(base_rain)
    normal_gdd = sum(base_gdd) / len(base_gdd)
    rain_dep = departure_pct(rain, normal_rain)
    gdd_dep = departure_pct(gdd, normal_gdd)

    notes = []
    if len(baseline_years) < WMO_NORMAL_YEARS:
        notes.append(
            f"Normal is based on {len(baseline_years)} seasons; WMO normals use {WMO_NORMAL_YEARS}."
        )
    if coverage < 1:
        notes.append(f"Season totals computed from {coverage:.0%} of days; gaps count as zero.")
    if rain_dep is None:
        notes.append("Normal rainfall is zero, so departure is undefined.")
    if request.source == "era5":
        notes.append("ERA5 days are UTC calendar days.")

    return WeatherAssessment(
        source=request.source,
        latitude=request.latitude,
        longitude=request.longitude,
        grid_latitude=df.attrs["grid_latitude"],
        grid_longitude=df.attrs["grid_longitude"],
        season_start=request.season_start,
        season_end=request.season_end,
        crop=request.crop,
        rainfall=RainfallAssessment(
            actual_mm=round(rain, 1),
            normal_mm=round(normal_rain, 1),
            departure_pct=None if rain_dep is None else round(rain_dep, 1),
            deficit_pct=None if rain_dep is None else round(rainfall_deficit_pct(rain_dep), 1),
            category=imd_rainfall_category(rain_dep),
        ),
        gdu=GDUAssessment(
            tbase_c=tbase,
            actual=round(gdd, 1),
            normal=round(normal_gdd, 1),
            deviation=round(gdd - normal_gdd, 1),
            deviation_pct=None if gdd_dep is None else round(gdd_dep, 1),
        ),
        baseline_years=baseline_years,
        data_coverage_pct=round(coverage * 100, 1),
        notes=notes,
    )


def weather_agent(state: Mapping[str, Any]) -> dict[str, Any]:
    """LangGraph node: `weather_request` in, `weather` (WeatherAssessment) out."""
    request = WeatherRequest.model_validate(state["weather_request"])
    return {"weather": assess_weather(request)}
