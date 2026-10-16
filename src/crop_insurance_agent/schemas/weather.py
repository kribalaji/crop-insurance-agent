"""Input and output schemas for weather_agent."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

WeatherSource = Literal["era5", "imd"]

# IMD rainfall categories by percentage departure from normal.
RainfallCategory = Literal[
    "large_excess",  # >= +60%
    "excess",  # +20% to +59%
    "normal",  # -19% to +19%
    "deficient",  # -20% to -59%
    "large_deficient",  # -60% to -99%
    "no_rain",  # -100%
]

# Commonly cited base temperatures (deg C) for the simple-average GDD method.
CROP_BASE_TEMPS_C: dict[str, float] = {
    "rice": 10.0,
    "maize": 10.0,
    "sorghum": 10.0,
    "wheat": 0.0,
    "cotton": 15.6,
}


class WeatherRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=360)
    season_start: date
    season_end: date
    crop: str
    source: WeatherSource = "era5"
    tbase_c: float | None = Field(
        default=None, description="GDD base temperature; defaults from CROP_BASE_TEMPS_C"
    )
    baseline_start_year: int | None = None
    baseline_end_year: int | None = None
    min_baseline_years: int = Field(default=5, ge=1)
    min_coverage: float = Field(default=0.9, gt=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> "WeatherRequest":
        if self.season_end < self.season_start:
            raise ValueError("season_end must not be before season_start")
        if (self.season_end - self.season_start).days >= 366:
            raise ValueError("season must be shorter than one year")
        if self.tbase_c is None:
            crop = self.crop.strip().lower()
            if crop not in CROP_BASE_TEMPS_C:
                raise ValueError(
                    f"no default base temperature for crop {self.crop!r}; pass tbase_c"
                )
            self.tbase_c = CROP_BASE_TEMPS_C[crop]
        return self


class RainfallAssessment(BaseModel):
    actual_mm: float
    normal_mm: float
    departure_pct: float | None = Field(description="(actual - normal) / normal * 100")
    deficit_pct: float | None = Field(description="max(0, -departure_pct)")
    category: RainfallCategory | None


class GDUAssessment(BaseModel):
    tbase_c: float
    actual: float = Field(description="Cumulative growing degree units over the season")
    normal: float
    deviation: float = Field(description="actual - normal")
    deviation_pct: float | None


class WeatherAssessment(BaseModel):
    source: WeatherSource
    latitude: float
    longitude: float
    grid_latitude: float
    grid_longitude: float
    season_start: date
    season_end: date
    crop: str
    rainfall: RainfallAssessment
    gdu: GDUAssessment
    baseline_years: list[int]
    data_coverage_pct: float
    notes: list[str] = Field(default_factory=list)
