"""Multi-agent crop insurance claims system built on public data."""

import argparse
from datetime import date


def main() -> None:
    """`crop-insurance-agent weather ...`: run weather_agent and print JSON."""
    from crop_insurance_agent.agents.weather_agent import assess_weather
    from crop_insurance_agent.schemas.weather import WeatherRequest

    parser = argparse.ArgumentParser(prog="crop-insurance-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("weather", help="seasonal rainfall deficit and GDU deviation")
    w.add_argument("--lat", type=float, required=True)
    w.add_argument("--lon", type=float, required=True)
    w.add_argument("--start", type=date.fromisoformat, required=True)
    w.add_argument("--end", type=date.fromisoformat, required=True)
    w.add_argument("--crop", required=True)
    w.add_argument("--source", choices=["era5", "imd"], default="era5")
    w.add_argument("--tbase", type=float)
    args = parser.parse_args()

    request = WeatherRequest(
        latitude=args.lat,
        longitude=args.lon,
        season_start=args.start,
        season_end=args.end,
        crop=args.crop,
        source=args.source,
        tbase_c=args.tbase,
    )
    print(assess_weather(request).model_dump_json(indent=2))
