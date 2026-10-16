"""Supervisor graph: one routing node dispatching to the four agents.

Routing is rule-based rather than LLM-chosen, so the order of a claim is fixed
and auditable: weather -> policy -> claim (pauses for human approval) ->
advisory. Every worker returns to the supervisor, which picks the next step from
what is already in the state.

Expected domain failures (no usable weather data, unknown policy) are recorded
in `error` and routed straight to advisory_agent, so the farmer still gets an
explanation. Anything else propagates.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from crop_insurance_agent.agents.advisory_agent import advisory_agent
from crop_insurance_agent.agents.claim_agent import claim_agent
from crop_insurance_agent.agents.policy_agent import policy_agent
from crop_insurance_agent.agents.weather_agent import weather_agent
from crop_insurance_agent.config import Settings
from crop_insurance_agent.graph.state import ClaimState
from crop_insurance_agent.rag.store import PolicyNotFoundError, PolicyStore
from crop_insurance_agent.weather.sources import WeatherDataError

# (state key the agent fills, node name), in pipeline order.
PIPELINE = (
    ("weather", "weather_agent"),
    ("policy", "policy_agent"),
    ("claim", "claim_agent"),
    ("advisory", "advisory_agent"),
)
WORKERS = tuple(node for _, node in PIPELINE)
RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (WeatherDataError, PolicyNotFoundError)


def next_step(state: Mapping[str, Any]) -> str:
    if state.get("error"):
        return "advisory_agent" if state.get("advisory") is None else END
    for key, node in PIPELINE:
        if state.get(key) is None:
            return node
    return END


def supervisor(state: ClaimState) -> dict[str, Any]:
    return {"next": next_step(state)}


def _guarded(name: str, node: Callable[[ClaimState], dict[str, Any]]):
    def run(state: ClaimState) -> dict[str, Any]:
        try:
            return node(state)
        except RECOVERABLE_ERRORS as exc:
            return {"error": {"agent": name, "kind": type(exc).__name__, "message": str(exc)}}

    return run


def build_graph(
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    store: PolicyStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the claim graph. A checkpointer is required for the approval pause;
    an in-memory one is used unless another is given."""
    nodes = {
        "weather_agent": partial(weather_agent, settings=settings),
        "policy_agent": partial(policy_agent, settings=settings, llm=llm, store=store),
        "claim_agent": claim_agent,
        "advisory_agent": partial(advisory_agent, settings=settings, llm=llm),
    }
    g = StateGraph(ClaimState)
    g.add_node("supervisor", supervisor)
    for name, node in nodes.items():
        g.add_node(name, _guarded(name, node))
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], [*WORKERS, END])
    return g.compile(checkpointer=checkpointer or InMemorySaver())
