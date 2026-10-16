"""FastAPI backend: submit claims, review them, and read their outcome.

Each claim is one LangGraph thread (thread id = claim id) persisted in SQLite,
so a claim paused for approval survives restarts. Runs execute synchronously
inside the request (FastAPI's threadpool), and a per-claim lock stops two
requests from advancing the same claim at once.

    uv run crop-insurance-api            # http://127.0.0.1:8000/docs
"""

import argparse
import logging
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from crop_insurance_agent.api.models import (
    ClaimRunView,
    ClaimStatus,
    ClaimSubmission,
    ClaimSummary,
    PendingReview,
    PolicyList,
)
from crop_insurance_agent.config import Settings, get_settings
from crop_insurance_agent.graph.checkpoint import sqlite_checkpointer
from crop_insurance_agent.graph.supervisor import build_graph
from crop_insurance_agent.observability import record_run_status, trace_config
from crop_insurance_agent.rag.store import PolicyStore
from crop_insurance_agent.schemas.claim import ApprovalDecision

log = logging.getLogger(__name__)


class ClaimRunError(Exception):
    """An unexpected error stopped a run; the claim is left `stalled`."""

    def __init__(self, claim_id: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.claim_id = claim_id
        self.cause = cause


def _status(snapshot: StateSnapshot) -> ClaimStatus:
    if snapshot.interrupts:
        return "awaiting_approval"
    if snapshot.values.get("status") in ("completed", "failed"):
        return snapshot.values["status"]
    return "stalled"


def _view(claim_id: str, snapshot: StateSnapshot) -> ClaimRunView:
    v = snapshot.values
    pending = None
    if snapshot.interrupts:
        payload = snapshot.interrupts[0].value
        pending = PendingReview(
            question=payload["question"],
            assumptions=payload["assumptions"],
            estimate=payload["estimate"],
        )
    return ClaimRunView(
        claim_id=claim_id,
        status=_status(snapshot),
        pending_review=pending,
        weather=v.get("weather"),
        policy=v.get("policy"),
        claim=v.get("claim"),
        advisory=v.get("advisory"),
        error=v.get("error"),
    )


def create_app(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    store: PolicyStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    deps: dict[str, Any] = {}
    locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        saver = sqlite_checkpointer(settings.checkpoint_path)
        policy_store = store or PolicyStore(settings.chroma_path)
        deps["store"] = policy_store
        deps["saver"] = saver
        deps["graph"] = build_graph(
            settings=settings, llm=llm, store=policy_store, checkpointer=saver
        )
        try:
            yield
        finally:
            saver.conn.close()

    app = FastAPI(
        title="Crop insurance claims",
        description="Weather, policy, claim and advisory agents behind a human approval step.",
        lifespan=lifespan,
    )

    def graph() -> CompiledStateGraph:
        return deps["graph"]

    def config(claim_id: str) -> dict[str, Any]:
        return trace_config(
            {"configurable": {"thread_id": claim_id}},
            settings,
            session_id=claim_id,
            tags=["api"],
        )

    def snapshot(claim_id: str) -> StateSnapshot:
        snap = graph().get_state({"configurable": {"thread_id": claim_id}})
        if not snap.values:
            raise HTTPException(404, f"claim {claim_id!r} not found")
        return snap

    def advance(claim_id: str, payload: Any) -> ClaimRunView:
        try:
            graph().invoke(payload, config(claim_id))
        except Exception as exc:
            log.exception("claim %s stalled", claim_id)
            record_run_status(claim_id, "stalled", str(exc), settings)
            raise ClaimRunError(claim_id, exc) from exc
        view = _view(claim_id, snapshot(claim_id))
        if view.status in ("completed", "failed"):
            message = view.error and view.error.get("message")
            record_run_status(claim_id, view.status, message, settings)
        return view

    @app.exception_handler(ClaimRunError)
    def _stalled(request: Request, exc: ClaimRunError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "claim_id": exc.claim_id,
                "status": "stalled",
                "detail": f"{type(exc.cause).__name__}: {exc}",
                "hint": f"fix the cause, then POST /claims/{exc.claim_id}/retry",
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/policies")
    def policies() -> PolicyList:
        s: PolicyStore = deps["store"]
        s.sync(settings.policies_dir)
        return PolicyList(policies=s.policy_ids())

    @app.post("/claims", status_code=201)
    def submit(sub: ClaimSubmission) -> ClaimRunView:
        """Run a claim until it pauses for approval (or ends early on an error)."""
        with locks[sub.claim_id]:
            if graph().get_state({"configurable": {"thread_id": sub.claim_id}}).values:
                raise HTTPException(409, f"claim {sub.claim_id!r} already exists")
            state = {
                "weather_request": sub.weather.model_dump(mode="json"),
                "policy_request": {"policy_id": sub.policy_id},
                "claim_request": {"claim_id": sub.claim_id, "area_ha": sub.area_ha},
                "language": sub.language,
            }
            return advance(sub.claim_id, state)

    def _thread_ids() -> Iterator[str]:
        seen: set[str] = set()
        for cp in deps["saver"].list(None):
            tid = cp.config["configurable"]["thread_id"]
            if tid not in seen:
                seen.add(tid)
                yield tid

    @app.get("/claims")
    def list_claims(status: ClaimStatus | None = None) -> list[ClaimSummary]:
        out = []
        for claim_id in sorted(_thread_ids()):
            view = _view(claim_id, snapshot(claim_id))
            if status and view.status != status:
                continue
            payout = view.claim.payout_amount if view.claim else None
            if payout is None and view.pending_review:
                payout = view.pending_review.estimate.payout_amount
            out.append(
                ClaimSummary(
                    claim_id=claim_id,
                    status=view.status,
                    policy_id=view.policy.policy_id if view.policy else None,
                    payout_amount=payout,
                    error=view.error and view.error.get("message"),
                )
            )
        return out

    @app.get("/claims/{claim_id}")
    def get_claim(claim_id: str) -> ClaimRunView:
        return _view(claim_id, snapshot(claim_id))

    @app.post("/claims/{claim_id}/decision")
    def decide(claim_id: str, decision: ApprovalDecision) -> ClaimRunView:
        """Approve or reject a claim that is awaiting approval, then finish the run."""
        with locks[claim_id]:
            if _status(snapshot(claim_id)) != "awaiting_approval":
                raise HTTPException(409, f"claim {claim_id!r} is not awaiting approval")
            return advance(claim_id, Command(resume=decision.model_dump()))

    @app.post("/claims/{claim_id}/retry")
    def retry(claim_id: str) -> ClaimRunView:
        """Resume a stalled claim from its last checkpoint (no step is repeated)."""
        with locks[claim_id]:
            if _status(snapshot(claim_id)) != "stalled":
                raise HTTPException(409, f"claim {claim_id!r} is not stalled")
            return advance(claim_id, None)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="crop-insurance-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=args.host, port=args.port)
