"""Run the graph and yield streaming output to the caller.

Two entry points:
- get_streaming_response: yields plain strings (final-answer tokens).
  Used by the CLI runner and any plain-text consumer.
- get_streaming_events: yields tagged dict events (thinking / tool_call /
  tool_result_segment / answer_token). Used by the Chainlit UI to render
  an ephemeral "Working..." step alongside the final answer.
"""

import json
import re
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from loguru import logger

from fingent.models.factory import extract_text_content
from fingent.orchestrator.graph import create_graph

_RESPONSE_NODES = {
    "agent_node",
    "simple_response_node",
    "finalize_node",
    "cache_check_node",
}


async def get_streaming_response(
    messages: str,
    customer_name: str = "Guest",
    conversation_id: str | None = None,
    callbacks: list | None = None,
) -> AsyncGenerator[str, None]:
    graph = create_graph()
    thread_id = conversation_id or "default-thread"
    actor_id = _sanitize_actor_id(customer_name)

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "customer_name": customer_name,
            "actor_id": actor_id,
        },
        # Default 25 is snug once the agent is iterating through tool returns
        # and parallel tool calls. Scale with MAX_TOOL_CALLS_PER_TURN so
        # a single question can actually USE the tool budget — at 16
        # tools a question can hit ~34 node transitions; 55 keeps headroom.
        "recursion_limit": 55,
    }
    if callbacks:
        config["callbacks"] = callbacks
    input_data = {
        "messages": [HumanMessage(content=messages)],
        "customer_name": customer_name,
        "tool_call_count": 0,
    }

    logger.info(
        f"Running QA graph (thread_id={thread_id}, actor_id={actor_id})"
    )

    current_node: str | None = None
    streamed_any = False
    final_state = None

    try:
        async for event in graph.astream_events(
            input=input_data, config=config, version="v2"
        ):
            event_type = event.get("event")
            event_data = event.get("data", {})
            name = event.get("name", "")

            if event_type == "on_chain_start" and name in _RESPONSE_NODES:
                current_node = name
                logger.debug(f"Streaming: entered response node '{name}'")

            elif event_type == "on_chain_end":
                output = event_data.get("output")
                if output and isinstance(output, dict) and "messages" in output:
                    final_state = output
                if name == current_node:
                    current_node = None

            elif event_type == "on_chat_model_stream":
                if current_node not in _RESPONSE_NODES:
                    continue

                chunk = event_data.get("chunk")
                if not chunk or not isinstance(chunk, AIMessageChunk):
                    continue

                if chunk.tool_calls or chunk.tool_call_chunks:
                    continue

                content = extract_text_content(chunk.content)
                if content:
                    streamed_any = True
                    yield content

        if not streamed_any and final_state:
            logger.warning("No tokens streamed; extracting final answer from state")
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    text = extract_text_content(msg.content)
                    if text:
                        yield text
                        break
    except Exception as e:
        logger.error(f"Streaming error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        raise


def _sanitize_actor_id(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9\-_ ]", "", name)
    sanitized = sanitized.replace(" ", "-").lower()
    return f"user:{sanitized or 'guest'}"


# ---------------------------------------------------------------------------
# Tagged-event stream for the Chainlit UI
# ---------------------------------------------------------------------------

# Tool args we surface in the UI. Anything else (cache flags, internal
# config) is omitted to keep the step display clean.
_VISIBLE_TOOL_ARG_KEYS = {"question", "doc_id", "query"}


def _sanitize_tool_args(raw: Any) -> dict:
    """Pull out only user-facing tool args. Drops InjectedToolArg fields,
    runtime config, falsy values, and the literal string "None" (which
    DeepSeek and other models sometimes emit instead of leaving an
    optional arg unset)."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        if k not in _VISIBLE_TOOL_ARG_KEYS:
            continue
        if not v:
            continue
        if isinstance(v, str) and v.strip().lower() in ("none", "null"):
            continue
        out[k] = v
    return out


async def get_streaming_events(
    messages: str,
    customer_name: str = "Guest",
    conversation_id: str | None = None,
    callbacks: list | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream tagged events for the Chainlit UI.

    Yields dicts with one of these `kind` values:
      - "answer_token"          {kind, text}
      - "rewind_to_thinking"    {kind, text}
      - "tool_call"             {kind, tool, args}
      - "tool_result_segment"   {kind, doc_id, score, content}
    """
    graph = create_graph()
    thread_id = conversation_id or "default-thread"
    actor_id = _sanitize_actor_id(customer_name)

    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "customer_name": customer_name,
            "actor_id": actor_id,
        },
        "recursion_limit": 55,
    }
    if callbacks:
        config["callbacks"] = callbacks

    input_data = {
        "messages": [HumanMessage(content=messages)],
        "customer_name": customer_name,
        "tool_call_count": 0,
    }

    logger.info(
        f"Streaming events for thread_id={thread_id}, actor_id={actor_id}"
    )

    streamed_in_invocation: list[str] = []
    in_response_node = False

    try:
        async for event in graph.astream_events(
            input=input_data, config=config, version="v2"
        ):
            event_type = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            if event_type == "on_chain_start" and name in _RESPONSE_NODES:
                in_response_node = True
                streamed_in_invocation = []

            elif event_type == "on_chain_end" and name in _RESPONSE_NODES:
                in_response_node = False
                output = data.get("output")
                last_msg = _last_ai_message(output)
                has_tool_calls = bool(getattr(last_msg, "tool_calls", None)) if last_msg else False
                if has_tool_calls and streamed_in_invocation:
                    yield {
                        "kind": "rewind_to_thinking",
                        "text": "".join(streamed_in_invocation),
                    }
                streamed_in_invocation = []

            elif event_type == "on_chat_model_stream":
                if not in_response_node:
                    continue
                chunk = data.get("chunk")
                if not chunk or not isinstance(chunk, AIMessageChunk):
                    continue
                if chunk.tool_calls or chunk.tool_call_chunks:
                    if streamed_in_invocation:
                        yield {
                            "kind": "rewind_to_thinking",
                            "text": "".join(streamed_in_invocation),
                        }
                        streamed_in_invocation = []
                    continue
                content = extract_text_content(chunk.content)
                if content:
                    streamed_in_invocation.append(content)
                    yield {"kind": "answer_token", "text": content}

            elif event_type == "on_tool_start":
                raw_input = data.get("input") or {}
                args = _sanitize_tool_args(raw_input)
                yield {"kind": "tool_call", "tool": name, "args": args}

            elif event_type == "on_tool_end" and name == "search_kb":
                # search_kb returns a JSON string of [{score, doc_id, content}, ...]
                # (or {"error": "..."} on failure). LangGraph wraps the
                # return in a ToolMessage; pull the JSON string out of
                # .content. Parse and emit one event per segment so the
                # UI can build a Sources panel.
                output = data.get("output")
                if hasattr(output, "content"):
                    raw = output.content
                else:
                    raw = output
                if not isinstance(raw, str):
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, list):
                    continue
                for seg in parsed:
                    if not isinstance(seg, dict):
                        continue
                    content = seg.get("content") or ""
                    if not content:
                        continue
                    yield {
                        "kind": "tool_result_segment",
                        "doc_id": seg.get("doc_id", ""),
                        "score": seg.get("score"),
                        "content": content,
                    }

    except Exception as e:
        logger.error(f"Event streaming error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        raise


def _last_ai_message(state_or_output: Any) -> AIMessage | None:
    """Pull the most recent AIMessage from a graph state/output dict."""
    if not state_or_output or not isinstance(state_or_output, dict):
        return None
    messages = state_or_output.get("messages")
    if not messages:
        return None
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None
    if isinstance(messages, AIMessage):
        return messages
    return None
