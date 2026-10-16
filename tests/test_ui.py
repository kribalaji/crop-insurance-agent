"""Streamlit UI driven by AppTest against the real FastAPI app (stub LLM)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from crop_insurance_agent.api import create_app
from crop_insurance_agent.config import Settings
from crop_insurance_agent.rag import PolicyStore
from crop_insurance_agent.schemas.advisory import AdvisoryDraft
from crop_insurance_agent.schemas.policy import PolicyTerms
from crop_insurance_agent.ui.client import ApiClient, ApiError
from tests.conftest import hash_embedder
from tests.factories import StubLLM
from tests.test_api import _submission
from tests.test_graph import ADVICE, TERMS

APP = str(Path(__file__).parents[1] / "src/crop_insurance_agent/ui/app.py")


@pytest.fixture
def terms() -> dict:
    """Mutable holder so a test can change what the stub policy LLM returns."""
    return {"value": TERMS}


@pytest.fixture
def client(data_dir: Path, policies_dir: Path, tmp_path: Path, terms: dict):
    settings = Settings(
        _env_file=None, data_dir=data_dir, policies_dir=policies_dir, cache_dir=tmp_path
    )
    app = create_app(
        settings,
        llm=StubLLM({PolicyTerms: lambda _: terms["value"], AdvisoryDraft: ADVICE}),
        store=PolicyStore(settings.chroma_path, embedder=hash_embedder),
    )
    with TestClient(app) as http:
        yield ApiClient(http)


def _app(client: ApiClient) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["api"] = client
    return at.run()


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.subheader]
    parts += [m.value for m in at.success]
    parts += (
        [m.value for m in at.info] + [m.value for m in at.warning] + [m.value for m in at.error]
    )
    parts += [f"{m.label}={m.value}" for m in at.metric]
    return "\n".join(str(p) for p in parts)


def test_reviewer_approves_from_the_queue(client):
    client.submit(_submission("C-1"))
    at = _app(client)
    assert not at.exception
    assert "Estimated payout=Rs 54,000" in _texts(at)

    at.text_input(key="reviewer-C-1").input("officer-1")
    at.text_input(key="comment-C-1").input("checked")
    next(b for b in at.button if b.label == "Approve").click()
    at.run()
    assert "Claim C-1 🟢 completed" in _texts(at)

    claim = client.claim("C-1")["claim"]
    assert claim["status"] == "approved" and claim["approval"]["comment"] == "checked"


def test_reviewer_name_is_required(client):
    client.submit(_submission("C-2"))
    at = _app(client)
    next(b for b in at.button if b.label == "Reject").click()
    at.run()
    assert "Enter the reviewer's name." in _texts(at)
    assert client.claim("C-2")["status"] == "awaiting_approval"


def test_default_assumptions_are_shown_to_the_reviewer(client, terms):
    no_exit = TERMS.triggers[1].model_copy(update={"exit_threshold": None})
    terms["value"] = TERMS.model_copy(update={"triggers": [no_exit]})
    client.submit(_submission("C-6"))
    at = _app(client)
    warnings = [w.value for w in at.warning]
    assert any(w.startswith("DEFAULT ASSUMPTION, not from the policy") for w in warnings)


def test_failed_claim_is_flagged_in_all_claims(client):
    client.submit(_submission("C-3", latitude=20.0, longitude=85.0))
    at = _app(client)
    assert not at.exception
    text = _texts(at)
    assert "No claims are waiting for approval." in text
    assert "Run failed at **weather_agent**" in text
    assert "🔴 failed" in text


def test_submit_form_runs_a_claim(client):
    at = _app(client)
    form_inputs = {t.label: t for t in at.text_input}
    form_inputs["Claim id (blank for automatic)"].input("C-4")
    numbers = {n.label: n for n in at.number_input}
    numbers["Latitude"].set_value(11.76)
    numbers["Longitude"].set_value(79.24)
    next(b for b in at.button if b.label == "Submit claim").click()
    at.run()
    assert not at.exception
    assert "Claim C-4: 🟡 awaiting approval" in _texts(at)


def test_client_surfaces_stalled_hint():
    import httpx

    def handler(request):
        return httpx.Response(
            500, json={"claim_id": "X", "status": "stalled", "detail": "boom", "hint": "retry"}
        )

    api = ApiClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t"))
    with pytest.raises(ApiError) as exc:
        api.retry("X")
    assert exc.value.detail["hint"] == "retry"
