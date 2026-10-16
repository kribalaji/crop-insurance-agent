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
    c = sub.add_parser(
        "claim", parents=[weather_args], help="run the full graph with interactive approval"
    )
    c.add_argument("--policy-id", required=True, help="policy PDF file name without .pdf")
    c.add_argument("--area", type=float, required=True, help="insured area in hectares")
    c.add_argument("--claim-id", default="cli-claim")
    c.add_argument("--reviewer", required=True, help="name recorded on the approval")
    sub.add_parser("ingest-policies", help="index the PDFs in POLICIES_DIR into Chroma")
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "claim":
        _run_claim(args)
        return

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


def _run_claim(args: argparse.Namespace) -> None:
    import json

    from langgraph.types import Command

    from crop_insurance_agent.graph import build_graph
    from crop_insurance_agent.observability import flush, record_run_status, trace_config

    graph = build_graph()
    config = trace_config(
        {"configurable": {"thread_id": args.claim_id}},
        session_id=args.claim_id,
        tags=["cli"],
    )
    state = graph.invoke(
        {
            "weather_request": {
                "latitude": args.lat,
                "longitude": args.lon,
                "season_start": args.start.isoformat(),
                "season_end": args.end.isoformat(),
                "crop": args.crop,
                "source": args.source,
                "tbase_c": args.tbase,
            },
            "policy_request": {"policy_id": args.policy_id},
            "claim_request": {"claim_id": args.claim_id, "area_ha": args.area},
        },
        config,
    )
    if pending := state.get("__interrupt__"):
        review = pending[0].value
        print(json.dumps(review["estimate"], indent=2))
        for a in review["assumptions"]:
            print(f"\n!! {a}")
        answer = input(f"\n{review['question']} [y/N] ").strip().lower()
        comment = input("Comment (optional): ").strip() or None
        decision = {"approved": answer == "y", "reviewer": args.reviewer, "comment": comment}
        state = graph.invoke(Command(resume=decision), config)

    if state.get("claim") is not None:
        claim = state["claim"]
        print(f"\nClaim {claim.status}: Rs {claim.payout_amount:,.2f}")
    if state.get("error"):
        print(f"\nCould not assess claim ({state['error']['agent']}): {state['error']['message']}")
    advisory = state["advisory"]
    print(f"\n{advisory.headline}\n{advisory.explanation}")
    if advisory.generated_by == "template":
        print("(template wording: the LLM text failed the number check)")
    error = state.get("error")
    record_run_status(args.claim_id, state["status"], error and error["message"])
    flush()
    if state["status"] == "failed":
        raise SystemExit(f"\nRun FAILED at {error['agent']}: {error['message']}")
