"""Graph edges (conditional routers)."""

from typing import Literal

from langchain_core.messages import AIMessage
from loguru import logger

from fingent.orchestrator.state import AgentState

MAX_TOOL_CALLS_PER_TURN = 16


def route_by_intent(state: AgentState) -> Literal["cache_check", "simple_response"]:
    """Route router output: rag_query → cache_check, otherwise → simple_response."""
    intent = state.get("intent", "rag_query")
    if intent == "rag_query":
        return "cache_check"
    return "simple_response"


def route_after_cache(state: AgentState) -> Literal["agent"]:
    """Always route to agent after cache check.

    On a hit the cached answer is injected as context for the agent to
    evaluate. On a miss the agent proceeds with normal RAG.
    """
    return "agent"


def should_continue(state: AgentState) -> Literal["tools", "finalize", "end"]:
    """ReAct loop decision:
    - last message has tool_calls and prior count under cap → run tools
    - last message has tool_calls but PRIOR count >= cap → finalize
    - last message is a plain AI text response → end

    Cap semantics: the cap stops further loop iterations, not the current
    batch of tool calls. We compare `tool_call_count - len(last.tool_calls)`
    (count *before* this iteration's emission) against the cap. A single
    oversized fan-out (e.g., 12 calls when prior count was 0) therefore
    executes in full instead of being aborted with zero executed calls —
    the cap kicks in on the NEXT iteration if the agent tries to fan out
    again.
    """
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)

    if not messages:
        return "end"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        prior_count = tool_call_count - len(last.tool_calls)
        if prior_count >= MAX_TOOL_CALLS_PER_TURN:
            logger.warning(
                f"Tool call limit ({MAX_TOOL_CALLS_PER_TURN}) reached "
                f"(prior_count={prior_count}); routing to finalize"
            )
            return "finalize"
        return "tools"
    return "end"
