"""IMD binary (.grd) archives read through imdlib."""

from pathlib import Path

import numpy as np
import pytest

from crop_insurance_agent.weather.sources import (
    WeatherDataError,
    load_point_daily,
    source_files,
)

YEAR = 2023  # 365 days
RAIN_LATS = np.arange(6.5, 38.5 + 0.125, 0.25)  # 129
RAIN_LONS = np.arange(66.5, 100.0 + 0.125, 0.25)  # 135
TEMP_LATS = np.arange(7.5, 37.5 + 0.5, 1.0)  # 31
TEMP_LONS = np.arange(67.5, 97.5 + 0.5, 1.0)  # 31


def _write_grd(folder: Path, value: float, lats, lons) -> np.ndarray:
    data = np.full((365, len(lats), len(lons)), value, dtype="<f4")
    folder.mkdir(parents=True, exist_ok=True)
    return data


@pytest.fixture(scope="module")
def imd_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data") / "imd"
    rain = _write_grd(root / "rain", 2.4, RAIN_LATS, RAIN_LONS)
    rain[:, *_idx(RAIN_LATS, RAIN_LONS, 8.0, 68.0)] = -999.0  # outside the land mask
    rain[10, *_idx(RAIN_LATS, RAIN_LONS, 11.75, 79.25)] = -999.0
    rain.tofile(root / "rain" / f"Rainfall_ind{YEAR}_rfp25.grd")

    tmax = _write_grd(root / "tmax", 30.0, TEMP_LATS, TEMP_LONS)
    tmax[20, *_idx(TEMP_LATS, TEMP_LONS, 11.5, 79.5)] = 99.9
    tmax.tofile(root / "tmax" / f"Maxtemp_MaxT_{YEAR}.GRD")
    # Year-named files are accepted too (imdlib's fn_format="yearwise").
    _write_grd(root / "tmin", 20.0, TEMP_LATS, TEMP_LONS).tofile(root / "tmin" / f"{YEAR}.GRD")
    return root


def _idx(lats, lons, lat, lon) -> tuple[int, int]:
    return int(np.argmin(abs(lats - lat))), int(np.argmin(abs(lons - lon)))


def test_grd_files_are_discovered(imd_dir: Path):
    names = [p.name for p in source_files(imd_dir.parent, "imd")]
    assert names == [f"Rainfall_ind{YEAR}_rfp25.grd", f"Maxtemp_MaxT_{YEAR}.GRD", f"{YEAR}.GRD"]


def test_grd_point_series(imd_dir: Path):
    df = load_point_daily(source_files(imd_dir.parent, "imd"), "imd", 11.76, 79.24)

    assert df.attrs == {"grid_latitude": 11.75, "grid_longitude": 79.25}
    assert len(df) == 365
    assert df.index[0].isoformat() == f"{YEAR}-01-01T00:00:00"
    assert df["precip_mm"].isna().sum() == 1  # -999 masked, not summed as rain
    assert df["tmax_c"].isna().sum() == 1  # 99.9 masked
    assert df["precip_mm"].sum() == pytest.approx(364 * 2.4, rel=1e-5)
    assert df["tmin_c"].eq(20.0).all()


def test_grd_outside_land_mask_has_no_rain(imd_dir: Path):
    with pytest.raises(WeatherDataError, match="no values"):
        load_point_daily(source_files(imd_dir.parent, "imd"), "imd", 8.0, 68.0)


def test_misplaced_grd_file_is_rejected(tmp_path: Path):
    bad = tmp_path / "imd" / "misc" / "Rainfall_ind2023_rfp25.grd"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"")
    with pytest.raises(WeatherDataError, match="imd/rain"):
        load_point_daily([bad], "imd", 11.75, 79.25)
