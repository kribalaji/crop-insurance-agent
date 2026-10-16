# crop-insurance-agent

Multi-agent crop insurance claims system (LangGraph supervisor) built only on public data.

> Work in progress: so far only `llm.py` and `weather_agent` are implemented.
> The architecture diagram and full docs come with the remaining agents.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in keys
uv run pytest
```

## Weather data

Put NetCDF files under `$DATA_DIR`:

- `era5/**/*.nc`: ERA5 / ERA5-Land, hourly or daily, with `tp` and `mx2t`/`mn2t` or `t2m`
- `imd/**/*.nc`: IMD gridded daily rainfall (`rf`/`RAINFALL`) and `TMAX`/`TMIN`

```bash
uv run crop-insurance-agent weather --lat 10.79 --lon 79.14 \
  --start 2023-06-01 --end 2023-09-30 --crop rice --source era5
```
