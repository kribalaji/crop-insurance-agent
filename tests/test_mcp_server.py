"""MCP policy-lookup server, exercised through an in-process MCP client."""

from pathlib import Path

import pytest
from mcp import Client

from crop_insurance_agent.config import Settings
from crop_insurance_agent.mcp_server import create_server
from crop_insurance_agent.rag import PolicyStore
from tests.conftest import hash_embedder
from tests.factories import POLICY

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server(policies_dir: Path, tmp_path: Path):
    settings = Settings(_env_file=None, policies_dir=policies_dir, cache_dir=tmp_path)
    return create_server(settings, PolicyStore(settings.chroma_path, embedder=hash_embedder))


async def test_tools_are_advertised(server):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"list_policies", "lookup_policy"}
    props = tools["lookup_policy"].input_schema["properties"]
    assert {"policy_id", "query", "k"} <= set(props)


async def test_list_policies(server):
    async with Client(server) as client:
        result = await client.call_tool("list_policies", {})
    assert not result.is_error
    assert result.structured_content["policies"] == [
        "sample_pmfby_mid_season_adversity",
        POLICY,
    ]


async def test_lookup_returns_cited_excerpts(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "lookup_policy",
            {"policy_id": POLICY, "query": "growing degree days high temperature", "k": 2},
        )
    assert not result.is_error
    excerpts = result.structured_content["excerpts"]
    assert 1 <= len(excerpts) <= 2
    assert all(e["policy_id"] == POLICY and e["page"] in (1, 2) for e in excerpts)
    assert any("growing degree days" in e["text"] for e in excerpts)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"policy_id": "nope", "query": "rain"}, "available: sample_pmfby"),
        ({"policy_id": POLICY, "query": "rain", "k": 50}, "between 1 and 10"),
    ],
)
async def test_bad_requests_are_tool_errors(server, args, message):
    async with Client(server) as client:
        result = await client.call_tool("lookup_policy", args)
    assert result.is_error
    assert message in result.content[0].text
