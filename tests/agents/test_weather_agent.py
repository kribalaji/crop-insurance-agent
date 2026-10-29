from datetime import date

import pandas as pd
import pytest

from crop_insurance_agent.agents import weather_agent as wa
from crop_insurance_agent.schemas.weather import WeatherAssessment, WeatherRequest
from crop_insurance_agent.weather import WeatherDataError, get_point_daily
from crop_insurance_agent.weather import cache as cache_mod
from tests.conftest import DAILY_RAIN_MM, POINT, TARGET_YEAR, YEARS

SEASON_DAYS = 122  # 1 Jun .. 30 Sep


def _request(**kw) -> WeatherRequest:
    base = dict(
        latitude=POINT[0],
        longitude=POINT[1],
        season_start=date(TARGET_YEAR, 6, 1),
        season_end=date(TARGET_YEAR, 9, 30),
        crop="rice",
    )
    return WeatherRequest(**(base | kw))


@pytest.mark.parametrize("source", ["era5", "imd"])
def test_assessment_matches_hand_calculation(settings, source):
    result = wa.assess_weather(_request(source=source), settings)

    assert (result.grid_latitude, result.grid_longitude) == (11.75, 79.25)
    assert result.baseline_years == [y for y in YEARS if y != TARGET_YEAR]
    assert result.data_coverage_pct == 100.0

    r = result.rainfall
    assert r.normal_mm == pytest.approx(SEASON_DAYS * DAILY_RAIN_MM, rel=1e-3)
    assert r.actual_mm == pytest.approx(SEASON_DAYS * DAILY_RAIN_MM / 2, rel=1e-3)
    assert r.departure_pct == pytest.approx(-50, abs=0.1)
    assert r.deficit_pct == pytest.approx(50, abs=0.1)
    assert r.category == "deficient"

    g = result.gdu
    assert g.tbase_c == 10.0
    assert g.normal == pytest.approx(SEASON_DAYS * 15, rel=1e-3)
    assert g.actual == pytest.approx(SEASON_DAYS * 17, rel=1e-3)
    assert g.deviation_pct == pytest.approx(13.3, abs=0.1)


def test_langgraph_node_contract(settings, monkeypatch):
    monkeypatch.setattr(wa, "get_settings", lambda: settings)
    out = wa.weather_agent({"weather_request": _request().model_dump(mode="json")})
    assert isinstance(out["weather"], WeatherAssessment)


def test_parquet_cache_is_reused(settings, monkeypatch):
    first = get_point_daily(settings.data_dir, settings.cache_dir, "era5", *POINT)
    assert len(list((settings.cache_dir / "weather").glob("*.parquet"))) == 1

    def boom(*a, **k):
        raise AssertionError("cache miss")

    monkeypatch.setattr(cache_mod, "load_point_daily", boom)
    second = get_point_daily(settings.data_dir, settings.cache_dir, "era5", *POINT)
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert second.attrs == {"grid_latitude": 11.75, "grid_longitude": 79.25}


def test_imd_missing_values_are_masked(settings):
    df = get_point_daily(settings.data_dir, settings.cache_dir, "imd", *POINT)
    assert df["tmax_c"].isna().sum() == 1
    with pytest.raises(WeatherDataError, match="no values"):
        get_point_daily(settings.data_dir, settings.cache_dir, "imd", 12.0, 79.0)


def test_point_outside_grid_is_rejected(settings):
    with pytest.raises(WeatherDataError, match="outside the grid"):
        wa.assess_weather(_request(latitude=20.0, longitude=85.0), settings)


def test_season_without_data_is_rejected(settings):
    with pytest.raises(WeatherDataError, match="complete days"):
        wa.assess_weather(
            _request(season_start=date(2030, 6, 1), season_end=date(2030, 9, 30)), settings
        )


def test_too_few_baseline_years_is_rejected(settings):
    with pytest.raises(WeatherDataError, match="baseline seasons"):
        wa.assess_weather(_request(min_baseline_years=30), settings)


def test_unknown_crop_needs_explicit_base_temperature():
    with pytest.raises(ValueError, match="tbase_c"):
        _request(crop="turmeric")
    assert _request(crop="turmeric", tbase_c=12.0).tbase_c == 12.0
