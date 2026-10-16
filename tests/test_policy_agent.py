from datetime import date
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from crop_insurance_agent.agents import policy_agent as pa
from crop_insurance_agent.config import Settings
from crop_insurance_agent.rag import PolicyStore
from crop_insurance_agent.schemas.policy import PolicyAssessment, PolicyRequest, PolicyTerms
from crop_insurance_agent.schemas.weather import RainfallAssessment
from tests.conftest import hash_embedder
from tests.factories import ABSOLUTE, HEAT, POLICY, RELATIVE, make_weather


def _terms(**kw) -> PolicyTerms:
    base = dict(
        crops=["Paddy (rice)"],
        risk_period_start=date(2023, 6, 1),
        risk_period_end=date(2023, 9, 30),
        sum_insured_per_ha=60000,
        triggers=[ABSOLUTE, RELATIVE, HEAT],
    )
    return PolicyTerms(**(base | kw))


class StubLLM:
    """Returns fixed PolicyTerms and records the prompt it was given."""

    def __init__(self, terms: PolicyTerms) -> None:
        self.terms = terms
        self.messages: list = []

    def with_structured_output(self, schema):
        assert schema is PolicyTerms

        def run(messages):
            self.messages = messages
            return self.terms

        return RunnableLambda(run)


@pytest.fixture
def policy_settings(policies_dir: Path, tmp_path: Path) -> Settings:
    return Settings(_env_file=None, policies_dir=policies_dir, cache_dir=tmp_path / "cache")


@pytest.fixture
def store(policy_settings: Settings) -> PolicyStore:
    return PolicyStore(policy_settings.chroma_path, embedder=hash_embedder)


def _assess(settings, store, terms, weather=None) -> tuple[PolicyAssessment, StubLLM]:
    llm = StubLLM(terms)
    result = pa.assess_policy(
        PolicyRequest(policy_id=POLICY), weather or make_weather(), settings, llm=llm, store=store
    )
    return result, llm


def test_triggers_are_evaluated_againstmake_weather(policy_settings, store):
    result, llm = _assess(policy_settings, store, _terms())

    assert result.crop_covered and result.triggered
    assert result.sum_insured_per_ha == 60000
    by_name = {e.trigger.name: e for e in result.evaluations}
    assert by_name[ABSOLUTE.name].observed == 292.8 and by_name[ABSOLUTE.name].met is True
    assert by_name[RELATIVE.name].observed == 50.0 and by_name[RELATIVE.name].met is True
    assert by_name[HEAT.name].observed == 13.3 and by_name[HEAT.name].met is False
    assert all(e.quote_verified for e in result.evaluations)  # quotes span PDF line wraps
    assert result.notes == []
    assert {c.policy_id for c in result.citations} == {POLICY}

    system, human = llm.messages
    assert "rainfall_deficit_pct" in system[1]
    assert "Claimed crop: rice" in human[1] and "[page 1]" in human[1]


def test_fabricated_quote_cannot_trigger(policy_settings, store):
    fake = ABSOLUTE.model_copy(update={"source_quote": "Cover applies below 900 mm."})
    result, _ = _assess(policy_settings, store, _terms(triggers=[fake]))
    (ev,) = result.evaluations
    assert ev.met is True and ev.quote_verified is False
    assert not result.triggered
    assert any("not in the policy text" in n for n in result.notes)


def test_uncovered_crop_does_not_trigger(policy_settings, store):
    result, _ = _assess(policy_settings, store, _terms(), make_weather(crop="wheat"))
    assert not result.crop_covered and not result.triggered
    assert any("not among the covered crops" in n for n in result.notes)


def test_undefined_metric_and_period_mismatch_are_reported(policy_settings, store):
    weather = make_weather(
        season_start=date(2023, 7, 1),
        rainfall=RainfallAssessment(
            actual_mm=0, normal_mm=0, departure_pct=None, deficit_pct=None, category=None
        ),
    )
    result, _ = _assess(policy_settings, store, _terms(triggers=[RELATIVE]), weather)
    assert result.evaluations[0].met is None and not result.triggered
    assert any("undefined" in n for n in result.notes)
    assert any("risk period" in n for n in result.notes)


def test_langgraph_node_contract(policy_settings, monkeypatch):
    monkeypatch.setattr(pa, "get_settings", lambda: policy_settings)
    monkeypatch.setattr(pa, "get_llm", lambda settings: StubLLM(_terms()))
    monkeypatch.setattr(pa, "PolicyStore", lambda path: PolicyStore(path, embedder=hash_embedder))
    out = pa.policy_agent(
        {
            "policy_request": {"policy_id": POLICY},
            "weather": make_weather().model_dump(mode="json"),
        }
    )
    assert isinstance(out["policy"], PolicyAssessment) and out["policy"].triggered
