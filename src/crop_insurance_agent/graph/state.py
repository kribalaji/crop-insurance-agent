"""Shared LangGraph state passed between the agents."""

from typing import Any, TypedDict


class ClaimState(TypedDict, total=False):
    weather_request: dict[str, Any]
    weather: Any  # WeatherAssessment
    policy_request: dict[str, Any]
    policy: Any  # PolicyAssessment
    claim_request: dict[str, Any]
    claim: Any  # ClaimDecision
