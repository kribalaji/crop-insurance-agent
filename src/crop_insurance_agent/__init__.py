"""Multi-agent crop insurance claims system built on public data."""

import argparse
from datetime import date


def main() -> None:
    """Command line entry point; each subcommand prints JSON."""
    import json

    from crop_insurance_agent.config import get_settings

    weather_args = argparse.ArgumentParser(add_help=False)
    weather_args.add_argument("--lat", type=float, required=True)
    weather_args.add_argument("--lon", type=float, required=True)
    weather_args.add_argument("--start", type=date.fromisoformat, required=True)
    weather_args.add_argument("--end", type=date.fromisoformat, required=True)
    weather_args.add_argument("--crop", required=True)
    weather_args.add_argument("--source", choices=["era5", "imd"], default="era5")
    weather_args.add_argument("--tbase", type=float)

    parser = argparse.ArgumentParser(prog="crop-insurance-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "weather", parents=[weather_args], help="seasonal rainfall deficit and GDU deviation"
    )
    p = sub.add_parser(
        "policy", parents=[weather_args], help="weather assessment checked against a policy"
    )
    p.add_argument("--policy-id", required=True, help="policy PDF file name without .pdf")
    sub.add_parser("ingest-policies", help="index the PDFs in POLICIES_DIR into Chroma")
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "ingest-policies":
        from crop_insurance_agent.rag import PolicyStore

        store = PolicyStore(settings.chroma_path)
        changed = store.sync(settings.policies_dir)
        print(json.dumps({"reindexed": changed, "indexed": store.policy_ids()}, indent=2))
        return

    from crop_insurance_agent.agents.weather_agent import assess_weather
    from crop_insurance_agent.schemas.weather import WeatherRequest

    weather = assess_weather(
        WeatherRequest(
            latitude=args.lat,
            longitude=args.lon,
            season_start=args.start,
            season_end=args.end,
            crop=args.crop,
            source=args.source,
            tbase_c=args.tbase,
        ),
        settings,
    )
    if args.command == "weather":
        print(weather.model_dump_json(indent=2))
        return

    from crop_insurance_agent.agents.policy_agent import assess_policy
    from crop_insurance_agent.schemas.policy import PolicyRequest

    policy = assess_policy(PolicyRequest(policy_id=args.policy_id), weather, settings)
    print(policy.model_dump_json(indent=2))
