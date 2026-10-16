"""Thin HTTP client for the FastAPI backend, used by the Streamlit UI."""

import os
from typing import Any

import httpx

# Claim runs call the LLM synchronously inside the request, so allow for slow providers.
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=300.0)


class ApiError(RuntimeError):
    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class ApiClient:
    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    @classmethod
    def from_env(cls) -> "ApiClient":
        base_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
        return cls(httpx.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT))

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = r.text
            detail = body
            if isinstance(body, dict) and body.get("status") != "stalled":
                detail = body.get("detail", body)  # stalled responses keep their hint
            raise ApiError(r.status_code, detail)
        return r.json()

    def policies(self) -> list[str]:
        return self._call("GET", "/policies")["policies"]

    def claims(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._call("GET", "/claims", params={"status": status} if status else None)

    def claim(self, claim_id: str) -> dict[str, Any]:
        return self._call("GET", f"/claims/{claim_id}")

    def submit(self, submission: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/claims", json=submission)

    def decide(
        self, claim_id: str, approved: bool, reviewer: str, comment: str | None
    ) -> dict[str, Any]:
        body = {"approved": approved, "reviewer": reviewer, "comment": comment or None}
        return self._call("POST", f"/claims/{claim_id}/decision", json=body)

    def retry(self, claim_id: str) -> dict[str, Any]:
        return self._call("POST", f"/claims/{claim_id}/retry")
