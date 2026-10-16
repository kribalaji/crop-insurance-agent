"""Textbook agro-meteorological formulas. Pure functions, no I/O."""

from datetime import date

import pandas as pd

from crop_insurance_agent.schemas.weather import RainfallCategory


def daily_gdd(tmax_c: pd.Series, tmin_c: pd.Series, tbase_c: float) -> pd.Series:
    """Growing degree days, simple-average method: max(0, (Tmax + Tmin) / 2 - Tbase)."""
    return ((tmax_c + tmin_c) / 2 - tbase_c).clip(lower=0)


def departure_pct(actual: float, normal: float) -> float | None:
    """Percentage departure from normal: (actual - normal) / normal * 100."""
    if normal == 0:
        return None
    return (actual - normal) / normal * 100


def rainfall_deficit_pct(departure: float | None) -> float | None:
    """Deficit is the negative part of the departure, reported as a positive number."""
    return None if departure is None else max(0.0, -departure)


def imd_rainfall_category(departure: float | None) -> RainfallCategory | None:
    """IMD classification of rainfall departure (percentages rounded to integers)."""
    if departure is None:
        return None
    d = round(departure)
    if d >= 60:
        return "large_excess"
    if d >= 20:
        return "excess"
    if d >= -19:
        return "normal"
    if d >= -59:
        return "deficient"
    if d >= -99:
        return "large_deficient"
    return "no_rain"


def shifted_window(start: date, end: date, years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The same calendar window moved by `years` (29 Feb maps to 28 Feb)."""
    offset = pd.DateOffset(years=years)
    return pd.Timestamp(start) + offset, pd.Timestamp(end) + offset
