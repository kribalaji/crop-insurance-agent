"""Shared LangGraph state passed between the agents."""

from typing import Any, TypedDict


class AgentError(TypedDict):
    agent: str
    kind: str
    message: str


class ClaimState(TypedDict, total=False):
    # Inputs
    weather_request: dict[str, Any]
    policy_request: dict[str, Any]
    claim_request: dict[str, Any]
    language: str  # advisory language code; "en" for now
    # Agent outputs
    weather: Any  # WeatherAssessment
    policy: Any  # PolicyAssessment
    claim: Any  # ClaimDecision
    advisory: Any  # AdvisoryNote
    # Supervisor bookkeeping
    next: str
    error: AgentError
    # Set when the run ends: "completed", or "failed" if an agent recorded an error
    # (the farmer still gets an advisory, but the run is not treated as a success).
    status: str
