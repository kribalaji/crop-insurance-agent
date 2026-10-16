"""Langfuse tracing for graph runs.

Tracing is on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, unless
LANGFUSE_ENABLED=false. `trace_config()` adds the Langfuse LangChain callback to
a LangGraph run config, so every node, retrieval and LLM call becomes a span.
Using the claim id as the session id groups a claim's first run and its
post-approval resume into one Langfuse session.

Traces include prompts and agent outputs (policy excerpts, farm location,
payout amounts). Disable tracing or self-host Langfuse if that is not acceptable.
"""

from typing import Any

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace.export import SpanExporter

from crop_insurance_agent.config import Settings, get_settings

_initialised: set[str] = set()


def tracing_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return s.langfuse_enabled and bool(s.langfuse_public_key and s.langfuse_secret_key)


def init_langfuse(settings: Settings, span_exporter: SpanExporter | None = None) -> Langfuse:
    """Create the Langfuse client for these credentials once per process."""
    assert settings.langfuse_public_key and settings.langfuse_secret_key
    public_key = settings.langfuse_public_key.get_secret_value()
    if public_key not in _initialised:
        Langfuse(
            public_key=public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            span_exporter=span_exporter,
        )
        _initialised.add(public_key)
    return get_client(public_key=public_key)


def trace_config(
    config: dict[str, Any],
    settings: Settings | None = None,
    *,
    session_id: str,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Return `config` with the Langfuse callback and session metadata added.

    Returns `config` unchanged when tracing is disabled.
    """
    settings = settings or get_settings()
    if not tracing_enabled(settings):
        return config
    init_langfuse(settings)
    assert settings.langfuse_public_key
    handler = CallbackHandler(public_key=settings.langfuse_public_key.get_secret_value())
    metadata = {
        **config.get("metadata", {}),
        "langfuse_session_id": session_id,
        "langfuse_tags": ["crop-insurance", *(tags or [])],
    }
    if user_id:
        metadata["langfuse_user_id"] = user_id
    return {
        **config,
        "callbacks": [*config.get("callbacks", []), handler],
        "metadata": metadata,
        "run_name": config.get("run_name", "crop-insurance-claim"),
    }


def record_run_status(
    session_id: str, status: str, comment: str | None = None, settings: Settings | None = None
) -> None:
    """Score the claim's Langfuse session with its final status (filterable in the UI)."""
    settings = settings or get_settings()
    if tracing_enabled(settings):
        init_langfuse(settings).create_score(
            name="run_status",
            value=status,
            session_id=session_id,
            data_type="CATEGORICAL",
            comment=comment,
        )


def flush(settings: Settings | None = None) -> None:
    """Send buffered spans; call before a short-lived process exits."""
    settings = settings or get_settings()
    if tracing_enabled(settings):
        init_langfuse(settings).flush()
