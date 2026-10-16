"""Langfuse tracing, captured with an in-memory OpenTelemetry exporter (no network)."""

from pathlib import Path

import pytest
from langgraph.types import Command
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from crop_insurance_agent.config import Settings
from crop_insurance_agent.graph import build_graph
from crop_insurance_agent.observability import flush, trace_config, tracing_enabled
from crop_insurance_agent.observability.tracing import init_langfuse
from crop_insurance_agent.rag import PolicyStore
from crop_insurance_agent.schemas.advisory import AdvisoryDraft
from crop_insurance_agent.schemas.policy import PolicyTerms
from tests.conftest import hash_embedder
from tests.factories import StubLLM
from tests.test_graph import ADVICE, TERMS, _inputs

KEYS = {"langfuse_public_key": "pk-lf-test", "langfuse_secret_key": "sk-lf-test"}


def test_disabled_without_keys_or_when_switched_off():
    assert not tracing_enabled(Settings(_env_file=None))
    assert not tracing_enabled(Settings(_env_file=None, langfuse_enabled=False, **KEYS))
    config = {"configurable": {"thread_id": "t"}}
    assert trace_config(config, Settings(_env_file=None), session_id="t") is config


@pytest.fixture(scope="module")
def exporter() -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    init_langfuse(Settings(_env_file=None, **KEYS), span_exporter=exp)
    return exp


def test_claim_run_is_traced_as_one_session(exporter, data_dir: Path, policies_dir: Path, tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=data_dir,
        policies_dir=policies_dir,
        cache_dir=tmp_path,
        **KEYS,
    )
    graph = build_graph(
        settings=settings,
        llm=StubLLM({PolicyTerms: TERMS, AdvisoryDraft: ADVICE}),
        store=PolicyStore(settings.chroma_path, embedder=hash_embedder),
    )
    config = trace_config(
        {"configurable": {"thread_id": "C-42"}}, settings, session_id="C-42", tags=["test"]
    )
    assert config["configurable"] == {"thread_id": "C-42"}
    assert config["metadata"]["langfuse_session_id"] == "C-42"

    graph.invoke(_inputs(), config)
    graph.invoke(Command(resume={"approved": True, "reviewer": "officer-1"}), config)
    flush(settings)

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"weather_agent", "policy_agent", "claim_agent", "advisory_agent"} <= names
    roots = [s for s in spans if s.name == "crop-insurance-claim"]
    assert len(roots) == 2  # first run and post-approval resume
    sessions = {s.attributes.get("session.id") for s in roots}
    assert sessions == {"C-42"}
