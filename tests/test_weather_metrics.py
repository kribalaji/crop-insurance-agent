from datetime import date

import pandas as pd
import pytest

from crop_insurance_agent.weather.metrics import (
    daily_gdd,
    departure_pct,
    imd_rainfall_category,
    rainfall_deficit_pct,
    shifted_window,
)


def test_daily_gdd_simple_average_clipped_at_zero():
    tmax = pd.Series([30.0, 12.0, 8.0])
    tmin = pd.Series([20.0, 6.0, 2.0])
    assert daily_gdd(tmax, tmin, 10.0).tolist() == [15.0, 0.0, 0.0]


def test_departure_and_deficit():
    assert departure_pct(50, 100) == -50
    assert departure_pct(130, 100) == pytest.approx(30)
    assert departure_pct(10, 0) is None
    assert rainfall_deficit_pct(-50.0) == 50.0
    assert rainfall_deficit_pct(30.0) == 0.0
    assert rainfall_deficit_pct(None) is None


@pytest.mark.parametrize(
    ("departure", "category"),
    [
        (60, "large_excess"),
        (59.4, "excess"),
        (20, "excess"),
        (19, "normal"),
        (-19, "normal"),
        (-19.6, "deficient"),
        (-59, "deficient"),
        (-60, "large_deficient"),
        (-99, "large_deficient"),
        (-100, "no_rain"),
        (None, None),
    ],
)
def test_imd_rainfall_category(departure, category):
    assert imd_rainfall_category(departure) == category


def test_shifted_window_handles_leap_day_and_year_crossing():
    assert shifted_window(date(2024, 2, 29), date(2024, 3, 31), -1) == (
        pd.Timestamp("2023-02-28"),
        pd.Timestamp("2023-03-31"),
    )
    assert shifted_window(date(2023, 10, 15), date(2024, 3, 31), -2) == (
        pd.Timestamp("2021-10-15"),
        pd.Timestamp("2022-03-31"),
    )
