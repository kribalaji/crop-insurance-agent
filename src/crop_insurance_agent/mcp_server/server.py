"""MCP server exposing policy lookup over the Chroma policy index.

Tools are retrieval only: they return verbatim policy excerpts with page
citations and never call an LLM, so the server needs no API key.

    uv run crop-insurance-mcp                          # stdio (Claude Desktop, IDEs)
    uv run crop-insurance-mcp --transport streamable-http --port 8001
"""

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from crop_insurance_agent.config import Settings, get_settings
from crop_insurance_agent.rag.store import PolicyNotFoundError, PolicyStore
from crop_insurance_agent.schemas.policy import Citation

INSTRUCTIONS = """\
Look up crop insurance policy documents. Call list_policies to see which policy ids
exist, then lookup_policy with a policy id and a natural-language question (for example
"deficit rainfall trigger and maximum payout"). Results are verbatim excerpts with page
numbers; quote them rather than paraphrasing when citing policy terms.
"""

MAX_EXCERPTS = 10


class PolicyList(BaseModel):
    policies: list[str] = Field(description="Policy ids accepted by lookup_policy")


class PolicyLookup(BaseModel):
    policy_id: str
    query: str
    excerpts: list[Citation]


def create_server(settings: Settings | None = None, store: PolicyStore | None = None) -> MCPServer:
    settings = settings or get_settings()
    state: dict[str, PolicyStore] = {}

    def get_store() -> PolicyStore:
        # Created on first use: loading the embedding model is slow.
        if "store" not in state:
            state["store"] = store or PolicyStore(settings.chroma_path)
        state["store"].sync(settings.policies_dir)
        return state["store"]

    server = MCPServer(name="crop-insurance-policies", instructions=INSTRUCTIONS)

    @server.tool(title="List policies")
    def list_policies() -> PolicyList:
        """List the ids of all indexed crop insurance policy documents."""
        return PolicyList(policies=get_store().policy_ids())

    @server.tool(title="Look up policy")
    def lookup_policy(policy_id: str, query: str, k: int = 4) -> PolicyLookup:
        """Search one policy document and return the most relevant excerpts with page numbers.

        Args:
            policy_id: A policy id from list_policies.
            query: What to look for, e.g. "deficit rainfall trigger" or "sum insured".
            k: Number of excerpts to return (1-10).
        """
        if not 1 <= k <= MAX_EXCERPTS:
            raise ToolError(f"k must be between 1 and {MAX_EXCERPTS}")
        s = get_store()
        try:
            chunks = s.search(policy_id, query, k=k)
        except PolicyNotFoundError as exc:
            raise ToolError(f"{exc}; available: {', '.join(s.policy_ids())}") from exc
        return PolicyLookup(
            policy_id=policy_id,
            query=query,
            excerpts=[Citation(policy_id=c.policy_id, page=c.page, text=c.text) for c in chunks],
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="crop-insurance-mcp")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = create_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
