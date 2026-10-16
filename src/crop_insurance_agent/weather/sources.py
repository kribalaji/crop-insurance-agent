"""Read gridded ERA5 / IMD NetCDF files into a daily series at one grid point.

Expected layout:
    $DATA_DIR/era5/**/*.nc   ERA5 / ERA5-Land, hourly or daily. Precipitation `tp`
                             (metres); temperature `mx2t`/`mn2t` or `t2m` (kelvin).
    $DATA_DIR/imd/**/*.nc    IMD gridded daily rainfall (mm) and Tmax/Tmin (deg C),
                             e.g. as written by imdlib's `to_netcdf`.

Variables may be split across files (e.g. one file per year per variable).
Days are UTC calendar days for ERA5 and IMD's own day definition for IMD.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

from crop_insurance_agent.schemas.weather import WeatherSource

COLUMNS = ["precip_mm", "tmax_c", "tmin_c"]

_TIME_NAMES = ("valid_time", "time", "TIME", "date")
_LAT_NAMES = ("latitude", "lat", "LATITUDE", "LAT")
_LON_NAMES = ("longitude", "lon", "LONGITUDE", "LON")


class WeatherDataError(RuntimeError):
    """Raised when weather data is missing, out of coverage or incomplete."""


@dataclass(frozen=True)
class _VarSpec:
    column: str
    candidates: tuple[str, ...]
    convert: Callable[[pd.Series], pd.Series]
    daily_agg: str
    # ERA5 accumulations / extremes are stamped at the end of their period, so the
    # 00:00 value belongs to the previous day.
    period_ending: bool = False


def _k_to_c(s: pd.Series) -> pd.Series:
    return s - 273.15


def _m_to_mm(s: pd.Series) -> pd.Series:
    return s * 1000.0


def _imd_rain(s: pd.Series) -> pd.Series:
    return s.where(s >= 0)  # IMD uses -999 for missing / outside land mask


def _imd_temp(s: pd.Series) -> pd.Series:
    return s.where((s > -90) & (s < 99))  # IMD uses 99.9 for missing


_SPECS: dict[WeatherSource, list[_VarSpec]] = {
    "era5": [
        _VarSpec("precip_mm", ("tp", "total_precipitation"), _m_to_mm, "sum", True),
        _VarSpec("tmax_c", ("mx2t",), _k_to_c, "max", True),
        _VarSpec("tmax_c", ("t2m", "2m_temperature"), _k_to_c, "max"),
        _VarSpec("tmin_c", ("mn2t",), _k_to_c, "min", True),
        _VarSpec("tmin_c", ("t2m", "2m_temperature"), _k_to_c, "min"),
    ],
    "imd": [
        _VarSpec("precip_mm", ("rf", "RAINFALL", "rainfall", "rain"), _imd_rain, "sum"),
        _VarSpec("tmax_c", ("TMAX", "tmax", "max_temp", "maxtemp"), _imd_temp, "max"),
        _VarSpec("tmin_c", ("TMIN", "tmin", "min_temp", "mintemp"), _imd_temp, "min"),
    ],
}

# Reject a nearest grid point further than this from the request (degrees).
MAX_GRID_DISTANCE_DEG = 0.5


def source_files(data_dir: Path, source: WeatherSource) -> list[Path]:
    root = data_dir / source
    files = sorted(p for p in root.rglob("*") if p.suffix in {".nc", ".nc4", ".netcdf"})
    if not files:
        raise WeatherDataError(f"no NetCDF files found under {root}")
    return files


def _first(names: Iterable[str], candidates: Iterable[str]) -> str | None:
    names = set(names)
    return next((c for c in candidates if c in names), None)


def _extract_point(
    ds: xr.Dataset, var: str, lat: float, lon: float
) -> tuple[pd.Series, float, float]:
    tdim = _first(ds[var].dims, _TIME_NAMES)
    latn = _first(ds[var].dims, _LAT_NAMES)
    lonn = _first(ds[var].dims, _LON_NAMES)
    if not (tdim and latn and lonn):
        raise WeatherDataError(f"cannot identify time/lat/lon dims of {var}: {ds[var].dims}")

    da = ds[var]
    # Collapse auxiliary dims such as ERA5's `expver` (one member is NaN) or `number`.
    extra = [d for d in da.dims if d not in (tdim, latn, lonn)]
    if extra:
        da = da.mean(dim=extra, skipna=True)

    if float(ds[lonn].max()) > 180 and lon < 0:
        lon = lon % 360
    pt = da.sel({latn: lat, lonn: lon}, method="nearest")
    glat, glon = float(pt[latn]), float(pt[lonn])
    if abs(glat - lat) > MAX_GRID_DISTANCE_DEG or abs(glon - lon) > MAX_GRID_DISTANCE_DEG:
        raise WeatherDataError(f"({lat}, {lon}) is outside the grid coverage of {var}")

    s = pt.drop_vars([c for c in pt.coords if c != tdim]).to_series()
    s.index = pd.DatetimeIndex(s.index).rename("date")
    return s.astype("float64"), glat, glon


def _to_daily(raw: pd.Series, spec: _VarSpec) -> pd.Series:
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    day = pd.Timedelta(days=1)
    step = pd.Series(raw.index).diff().median() if len(raw) > 1 else day
    if spec.period_ending and step < day:
        raw.index = raw.index - step
    grouped = raw.resample("D")
    daily = grouped.agg(spec.daily_agg)
    # A day with any missing value is missing, never zero rain or a partial extreme.
    expected = max(1, round(day / step))
    return daily.where(grouped.count() >= expected)


def load_point_daily(
    files: list[Path], source: WeatherSource, lat: float, lon: float
) -> pd.DataFrame:
    """Daily `precip_mm`, `tmax_c`, `tmin_c` at the grid point nearest (lat, lon).

    The returned frame's `attrs` carry `grid_latitude` and `grid_longitude`.
    """
    specs = _SPECS[source]
    raw: dict[int, list[pd.Series]] = {i: [] for i in range(len(specs))}
    grid: tuple[float, float] | None = None

    for path in files:
        with xr.open_dataset(path) as ds:
            for i, spec in enumerate(specs):
                var = _first(ds.data_vars, spec.candidates)
                if var is None:
                    continue
                s, glat, glon = _extract_point(ds, var, lat, lon)
                raw[i].append(spec.convert(s))
                grid = grid or (glat, glon)

    columns: dict[str, pd.Series] = {}
    for i, spec in enumerate(specs):
        # Earlier specs are preferred (e.g. ERA5 mx2t over t2m) when both exist.
        if raw[i] and spec.column not in columns:
            columns[spec.column] = _to_daily(pd.concat(raw[i]), spec)

    missing = [c for c in COLUMNS if c not in columns or columns[c].isna().all()]
    if missing or grid is None:
        raise WeatherDataError(f"{source} data at ({lat}, {lon}) has no values for {missing}")

    out = pd.DataFrame(columns)[COLUMNS].sort_index()
    out.index.name = "date"
    out.attrs = {"grid_latitude": grid[0], "grid_longitude": grid[1]}
    return out
