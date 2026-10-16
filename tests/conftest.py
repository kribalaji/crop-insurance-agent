"""Synthetic ERA5 / IMD NetCDF fixtures with hand-checkable answers.

Every season has a constant rain rate and a daily Tmax/Tmin of 30/20 deg C
(mean 25, so GDD at base 10 is 15/day). In TARGET_YEAR the Jun-Sep rain rate is
halved and temperatures are 2 deg C warmer (GDD 17/day).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from crop_insurance_agent.config import Settings

YEARS = range(2018, 2024)
TARGET_YEAR = 2023
LATS = [12.0, 11.75, 11.5]  # descending, as in ERA5
LONS = [79.0, 79.25, 79.5]
POINT = (11.76, 79.24)  # nearest grid point (11.75, 79.25)
DAILY_RAIN_MM = 2.4


def _in_target_season(times: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((times >= "2023-06-01") & (times < "2023-10-01"))


def _grid(values: np.ndarray) -> np.ndarray:
    return np.broadcast_to(values[:, None, None], (len(values), len(LATS), len(LONS))).copy()


def _write_era5(root: Path) -> None:
    # Hourly; tp is stamped at the end of each hour, so day D spans D 01:00 .. D+1 00:00.
    times = pd.date_range(f"{YEARS[0]}-01-01 01:00", f"{YEARS[-1] + 1}-01-01 00:00", freq="h")
    hour_start = times - pd.Timedelta(hours=1)
    target = _in_target_season(hour_start)
    tp = np.where(target, 0.5, 1.0) * DAILY_RAIN_MM / 24 / 1000  # metres per hour

    t2m_times = times - pd.Timedelta(hours=1)  # instantaneous, 00:00 .. 23:00
    phase = 2 * np.pi * t2m_times.hour.to_numpy() / 24
    t2m = 298.15 + 5 * np.sin(phase) + np.where(_in_target_season(t2m_times), 2.0, 0.0)

    coords = {"latitude": LATS, "longitude": LONS}
    dims = ("valid_time", "latitude", "longitude")
    era5 = root / "era5"
    era5.mkdir(parents=True)
    # Separate files per variable, as CDS downloads often are.
    xr.Dataset(
        {"tp": (dims, _grid(tp).astype("float32"))}, coords={"valid_time": times, **coords}
    ).to_netcdf(era5 / "tp.nc")
    xr.Dataset(
        {"t2m": (dims, _grid(t2m).astype("float32"))}, coords={"valid_time": t2m_times, **coords}
    ).to_netcdf(era5 / "t2m.nc")


def _write_imd(root: Path) -> None:
    times = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    target = _in_target_season(times)
    rf = np.where(target, 0.5, 1.0) * DAILY_RAIN_MM
    tmax = 30.0 + np.where(target, 2.0, 0.0)
    tmin = 20.0 + np.where(target, 2.0, 0.0)
    rf_grid, tmax_grid, tmin_grid = _grid(rf), _grid(tmax), _grid(tmin)
    rf_grid[:, 0, 0] = -999.0  # masked sea cell, as in IMD files
    tmax_grid[5, 1, 1] = 99.9  # a missing day outside every season window

    coords = {"time": times, "lat": LATS[::-1], "lon": LONS}
    dims = ("time", "lat", "lon")
    imd = root / "imd"
    imd.mkdir(parents=True)
    flip = slice(None, None, -1)
    xr.Dataset(
        {
            "rf": (dims, rf_grid[:, flip]),
            "TMAX": (dims, tmax_grid[:, flip]),
            "TMIN": (dims, tmin_grid[:, flip]),
        },
        coords=coords,
    ).to_netcdf(imd / "imd.nc")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    _write_era5(root)
    _write_imd(root)
    return root


@pytest.fixture
def settings(data_dir: Path, tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=data_dir, cache_dir=tmp_path / "cache")
