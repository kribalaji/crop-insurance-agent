"""Real LLM calls (policy extraction, advisory). Opt in: RUN_LIVE_LLM=1 uv run pytest -m live

Uses the configured LLM_PROVIDER and Chroma's default local embedding model.
"""

import os
from pathlib import Path

import pytest

from crop_insurance_agent.agents import policy_agent as pa
from crop_insurance_agent.config import Settings
from crop_insurance_agent.llm import get_llm
from crop_insurance_agent.rag import PolicyStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("RUN_LIVE_LLM"), reason="set RUN_LIVE_LLM=1"),
]


@pytest.fixture(scope="module")
def extracted(policies_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    settings = Settings(policies_dir=policies_dir, cache_dir=tmp_path_factory.mktemp("cache"))
    store = PolicyStore(settings.chroma_path)
    store.sync(policies_dir)
    llm = get_llm(settings=settings)

    def run(policy_id: str, crop: str):
        chunks = pa.retrieve(store, policy_id, crop)
        return pa.extract_terms(llm, policy_id, crop, chunks), chunks

    return run


def test_weather_index_term_sheet(extracted):
    terms, _ = extracted("sample_rwbcis_paddy_thanjavur", "rice")
    got = {(t.metric, t.comparator, t.threshold) for t in terms.triggers}
    assert got == {
        ("rainfall_actual_mm", "<", 600),
        ("rainfall_deficit_pct", ">=", 40),
        ("gdu_deviation_pct", ">", 15),
    }
    assert terms.sum_insured_per_ha == 60000


def test_yield_scheme_skips_unsupported_covers(extracted):
    terms, _ = extracted("sample_pmfby_mid_season_adversity", "rice")
    assert [(t.metric, t.comparator, t.threshold) for t in terms.triggers] == [
        ("rainfall_deficit_pct", ">=", 50)
    ]  # prevented sowing and CCE-based final claim are not weather triggers
    assert terms.sum_insured_per_ha == 55000


def test_advisory_numbers_come_from_facts():
    from crop_insurance_agent.agents.advisory_agent import (
        build_facts,
        unsupported_numbers,
        write_advisory,
    )
    from tests.factories import make_weather
    from tests.test_advisory_agent import _claim, _policy

    facts = build_facts(make_weather(), _policy(), _claim(), None)
    note = write_advisory(facts, get_llm())
    print(f"\n[{note.generated_by}] {note.headline}\n{note.explanation}\n{note.notes}")
    assert unsupported_numbers(f"{note.headline} {note.explanation}", facts) == []
    assert "48132" in (note.headline + note.explanation).replace(",", "")
