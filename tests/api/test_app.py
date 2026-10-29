"""FastAPI backend over the real graph with a SQLite checkpointer (stub LLM)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crop_insurance_agent.api import create_app
from crop_insurance_agent.config import Settings
from crop_insurance_agent.rag import PolicyStore
from crop_insurance_agent.schemas.advisory import AdvisoryDraft
from crop_insurance_agent.schemas.policy import PolicyTerms
from tests.conftest import hash_embedder
from tests.factories import ADVICE, APPROVE, POLICY, TERMS, StubLLM, api_submission


@pytest.fixture
def settings(data_dir: Path, policies_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, data_dir=data_dir, policies_dir=policies_dir, cache_dir=tmp_path
    )


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM({PolicyTerms: TERMS, AdvisoryDraft: ADVICE})


@pytest.fixture
def make_client(settings: Settings, llm: StubLLM):
    def make() -> TestClient:
        store = PolicyStore(settings.chroma_path, embedder=hash_embedder)
        return TestClient(create_app(settings, llm=llm, store=store))

    return make


def test_submit_review_and_complete(make_client):
    with make_client() as client:
        r = client.post("/claims", json=api_submission())
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "awaiting_approval"
        assert body["pending_review"]["estimate"]["payout_amount"] == 54000.0
        assert body["weather"]["rainfall"]["deficit_pct"] == 50.0
        assert body["claim"] is None and body["advisory"] is None

        r = client.post("/claims/C-1/decision", json=APPROVE)
        body = r.json()
        assert r.status_code == 200 and body["status"] == "completed"
        assert body["claim"]["payout_amount"] == 54000.0
        assert body["claim"]["approval"]["comment"] == "rain gauge agrees"
        assert body["advisory"]["headline"] == ADVICE.headline

        assert client.post("/claims/C-1/decision", json=APPROVE).status_code == 409
        assert client.get("/claims").json() == [
            {
                "claim_id": "C-1",
                "status": "completed",
                "policy_id": POLICY,
                "payout_amount": 54000.0,
                "error": None,
            }
        ]


def test_paused_claim_survives_restart(make_client):
    with make_client() as client:
        client.post("/claims", json=api_submission("C-2"))

    with make_client() as client:
        body = client.get("/claims/C-2").json()
        assert body["status"] == "awaiting_approval"
        assert body["pending_review"]["estimate"]["payout_amount"] == 54000.0
        assert body["policy"]["triggered"] is True  # pydantic state restored intact
        done = client.post("/claims/C-2/decision", json=APPROVE).json()
        assert done["status"] == "completed"


def test_agent_error_is_explained_and_listed_as_failed(make_client):
    with make_client() as client:
        r = client.post("/claims", json=api_submission("C-3", latitude=20.0, longitude=85.0))
        body = r.json()
        assert r.status_code == 201 and body["status"] == "failed"
        assert body["error"]["agent"] == "weather_agent"
        assert body["advisory"] is not None and body["claim"] is None

        failed = client.get("/claims", params={"status": "failed"}).json()
        assert [c["claim_id"] for c in failed] == ["C-3"]
        assert "outside the grid" in failed[0]["error"]
        assert client.get("/claims", params={"status": "completed"}).json() == []


def test_stalled_claim_resumes_without_repeating_steps(make_client, llm):
    outage = {"left": 1}

    def advice(messages):
        if outage["left"]:
            outage["left"] -= 1
            raise RuntimeError("LLM provider unavailable")
        return ADVICE

    llm.responses[AdvisoryDraft] = advice
    with make_client() as client:
        client.post("/claims", json=api_submission("C-4"))
        r = client.post("/claims/C-4/decision", json=APPROVE)
        assert r.status_code == 500
        assert r.json()["status"] == "stalled" and "retry" in r.json()["hint"]

        body = client.get("/claims/C-4").json()
        assert body["status"] == "stalled"
        assert body["claim"]["status"] == "approved"  # the decision was kept

        assert client.post("/claims/C-4/decision", json=APPROVE).status_code == 409
        done = client.post("/claims/C-4/retry").json()
        assert done["status"] == "completed" and done["advisory"] is not None
    assert [s for s, _ in llm.calls].count(PolicyTerms) == 1  # policy step not repeated


def test_request_errors(make_client):
    with make_client() as client:
        assert client.get("/claims/nope").status_code == 404
        client.post("/claims", json=api_submission("C-5"))
        assert client.post("/claims", json=api_submission("C-5")).status_code == 409
        r = client.post("/claims/C-5/decision", json={"approved": True})
        assert r.status_code == 422  # reviewer is required
        assert client.post("/claims/C-5/retry").status_code == 409
        bad = api_submission("bad id!")
        assert client.post("/claims", json=bad).status_code == 422


def test_policies_and_health(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert POLICY in client.get("/policies").json()["policies"]
