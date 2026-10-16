"""Parquet cache for point series extracted from gridded weather files.

Each cache file holds the full daily series for one source and point. Its name
includes a fingerprint of the source files (path, size, mtime), so adding or
replacing data files invalidates the cache automatically.
"""

import hashlib
from pathlib import Path

import pandas as pd

from crop_insurance_agent.schemas.weather import WeatherSource
from crop_insurance_agent.weather.sources import load_point_daily, source_files


def _fingerprint(files: list[Path], root: Path) -> str:
    h = hashlib.sha1()
    for p in files:
        st = p.stat()
        h.update(f"{p.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:12]


def cache_path(
    cache_dir: Path, source: WeatherSource, lat: float, lon: float, fingerprint: str
) -> Path:
    return cache_dir / "weather" / f"{source}_{lat:.3f}_{lon:.3f}_{fingerprint}.parquet"


def get_point_daily(
    data_dir: Path, cache_dir: Path, source: WeatherSource, lat: float, lon: float
) -> pd.DataFrame:
    """Daily point series, read from the Parquet cache or built and cached."""
    files = source_files(data_dir, source)
    path = cache_path(cache_dir, source, lat, lon, _fingerprint(files, data_dir / source))
    if path.exists():
        return pd.read_parquet(path)  # attrs round-trip via pandas parquet metadata

    df = load_point_daily(files, source, lat, lon)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return df
