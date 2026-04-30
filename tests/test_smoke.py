"""Smoke tests for FinGent.

Two layers:

1. **Import / construction tests** — always run. No external state required.
   Verifies the package layout, public API surface, default tool list, and
   graph compilation.

2. **End-to-end test** — opt-in via env vars. Loads a real KB, runs the
   agent against a known-answer question, asserts the answer contains the
   expected substring. Set:

       export FINGENT_TEST_KB_ID=...
       export FINGENT_TEST_STORE_DIR=/path/to/dsrag_store
       export FINGENT_TEST_QUESTION='What was MGIC primary IIF at year-end 2025?'
       export FINGENT_TEST_EXPECT='303.1'
       export DEEPSEEK_API_KEY=...
       export AWS_ACCESS_KEY_ID=...
       export AWS_SECRET_ACCESS_KEY=...
       export AWS_REGION=us-east-1

   Then run: `uv run pytest -m e2e`

The end-to-end test is marked `pytest.mark.e2e` and requires `-m e2e` (or
no marker filter) to actually execute.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Layer 1: imports + construction (always run)
# ---------------------------------------------------------------------------


def test_public_api_imports() -> None:
    """The two public top-level imports must always work."""
    from fingent import Agent, build_kb  # noqa: F401


def test_internal_imports() -> None:
    """Sub-package public surfaces must resolve without circular-import errors."""
    from fingent.models import (  # noqa: F401
        UsageCollector,
        cost_usd,
        get_model,
        normalize_model_id,
    )
    from fingent.orchestrator import (  # noqa: F401
        create_graph,
        get_streaming_events,
        get_streaming_response,
        reset_graph,
    )
    from fingent.retrieval import (  # noqa: F401
        AUTO_QUERY_GUIDANCE,
        HybridKnowledgeBase,
        get_kb,
        get_search_queries,
        search_kb,
        set_kb_location,
        smart_rrf_alpha,
    )


def test_default_tool_is_search_kb() -> None:
    """The default tool set is exactly [search_kb] — public-name guarantee."""
    from fingent.retrieval.tool import get_default_tools

    tools = get_default_tools()
    assert len(tools) == 1
    assert tools[0].name == "search_kb"


def test_graph_compiles() -> None:
    """create_graph() returns a compiled graph; reset_graph() invalidates it."""
    from fingent.orchestrator.graph import create_graph, reset_graph

    reset_graph()
    g1 = create_graph()
    assert g1 is not None
    g2 = create_graph()
    assert g1 is g2  # cached singleton

    reset_graph()
    g3 = create_graph()
    assert g3 is not g1  # rebuilt after reset


def test_agent_rejects_missing_store_dir(tmp_path: Path) -> None:
    """Agent.__init__ raises FileNotFoundError on a non-existent store_dir."""
    from fingent import Agent

    missing = tmp_path / "no-such-store"
    with pytest.raises(FileNotFoundError):
        Agent(kb_id="never_built", store_dir=missing)


def test_pricing_normalize() -> None:
    """The pricing normalizer strips Bedrock prefixes / version suffixes."""
    from fingent.models.usage import cost_usd, normalize_model_id

    assert normalize_model_id(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ) == "claude-sonnet-4-5"
    assert normalize_model_id("deepseek-v4-flash") == "deepseek-v4-flash"

    # Cost arithmetic — flash at $0.14/1M input, $0.55/1M output.
    cost = cost_usd("deepseek-v4-flash", input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 0.14) < 1e-9


# ---------------------------------------------------------------------------
# Layer 2: end-to-end (opt-in via env vars)
# ---------------------------------------------------------------------------


def _e2e_env_ok() -> tuple[bool, str]:
    """Check whether the e2e test has the env it needs.

    Returns (ok, reason_if_skipped).
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return False, "DEEPSEEK_API_KEY not set"
    if not os.environ.get("FINGENT_TEST_STORE_DIR"):
        return False, "FINGENT_TEST_STORE_DIR not set (point at an existing dsRAG store)"
    if not os.environ.get("FINGENT_TEST_KB_ID"):
        return False, "FINGENT_TEST_KB_ID not set"
    store_dir = Path(os.environ["FINGENT_TEST_STORE_DIR"])
    if not store_dir.exists():
        return False, f"FINGENT_TEST_STORE_DIR does not exist: {store_dir}"
    return True, ""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_end_to_end_against_real_kb() -> None:
    """Build an Agent against a real KB and verify a known-answer question.

    Skipped unless FINGENT_TEST_KB_ID + FINGENT_TEST_STORE_DIR + DEEPSEEK_API_KEY
    are all set.

    The default question + expected substring target a known fact in MGIC's
    2025 10-K (primary IIF = $303.1B). Override via FINGENT_TEST_QUESTION
    and FINGENT_TEST_EXPECT to point at any other corpus.
    """
    ok, reason = _e2e_env_ok()
    if not ok:
        pytest.skip(reason)

    from fingent import Agent

    agent = Agent(
        kb_id=os.environ["FINGENT_TEST_KB_ID"],
        store_dir=os.environ["FINGENT_TEST_STORE_DIR"],
    )

    question = os.environ.get(
        "FINGENT_TEST_QUESTION",
        "What was MGIC's primary insurance in force at December 31, 2025?",
    )
    expect = os.environ.get("FINGENT_TEST_EXPECT", "303.1")

    answer = await agent.ask(question)
    assert answer, "agent returned an empty answer"
    assert expect in answer, (
        f"expected substring {expect!r} not found in answer:\n{answer}"
    )
