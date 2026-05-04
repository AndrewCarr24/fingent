"""Chains for the router, the RAG agent, and the simple-response path.

Implements Bedrock prefix caching via `cachePoint` content blocks on the
system message and the last message of each ReAct turn. This is the
primary cost-saving mechanism — Bedrock caches the prefix at ~10% of
input-token price across ReAct iterations within a turn and across turns
within a session.

DeepSeek applies prefix caching server-side automatically; no client-
side markers are required.
"""

import os

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from fingent.models.factory import get_model, orchestrator_is_bedrock


def _agent_temp() -> float:
    """Orchestrator temperature. Production default 0.35; pinned to 0 in
    eval runs by setting FINGENT_AGENT_TEMPERATURE=0 before import."""
    return float(os.environ.get("FINGENT_AGENT_TEMPERATURE", "0.35"))
from fingent.orchestrator.catalog import format_for_prompt as format_catalog
from fingent.orchestrator.prompts import (
    AGENT_SYSTEM_PROMPT,
    ROUTER_PROMPT,
    SIMPLE_RESPONSE_PROMPT,
)
from fingent.retrieval.tool import get_default_tools

# Token budget for history sent to the agent LLM. Bounded by the
# reasoning model's context. 60K leaves comfortable headroom for the
# system prompt + tool calls/results within a multi-iteration ReAct
# loop, plus the model's output.
HISTORY_TOKEN_BUDGET = 60_000


# --- Tool registration -----------------------------------------------------
# The agent's tool list defaults to `[search_kb]` but can be overridden
# at Agent construction time via `extra_tools`. The override propagates
# through this module-level holder.
_active_tools: list | None = None


def set_active_tools(tools: list) -> None:
    """Set the tool list bound to the agent. Called by Agent.__init__."""
    global _active_tools
    _active_tools = list(tools)


def get_active_tools() -> list:
    """Return the active tool list (default = search_kb only)."""
    if _active_tools is None:
        return get_default_tools()
    return _active_tools


# --- History trimming + cache markers --------------------------------------

# finalize_node converts every ToolMessage in the trace into a synthesized
# HumanMessage prefixed with "[Tool result for '<name>']\n..." so the
# no-tools finalize chain doesn't trip Bedrock's toolConfig validation.
# A naive "anchor on the most-recent HumanMessage" rule would latch onto
# the last tool result, leaving the original user question evictable.
# Filter on the prefix to recognize synthesized HumanMessages.
_TOOL_RESULT_PREFIX = "[Tool result for '"


def _is_original_user_message(msg: BaseMessage) -> bool:
    """True iff `msg` is a HumanMessage produced by the user (or upstream
    runner), not one synthesized from a ToolMessage by finalize_node."""
    if not isinstance(msg, HumanMessage):
        return False
    content = msg.content
    if isinstance(content, str):
        return not content.startswith(_TOOL_RESULT_PREFIX)
    return True


def _turn_boundaries(messages: list[BaseMessage]) -> list[int]:
    """Indices of original HumanMessages — i.e., the start of each turn."""
    return [i for i, m in enumerate(messages) if _is_original_user_message(m)]


def _final_text_ai(
    messages: list[BaseMessage], start: int, end: int
) -> int | None:
    """Index of the last AIMessage in messages[start:end] that has no
    tool_calls and non-empty text content. None if the turn doesn't have
    one (interrupted, errored, or in flight)."""
    for i in range(end - 1, start - 1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and not m.tool_calls and m.content:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                return i
    return None


def _compact_completed_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replace each completed turn's body with just [Q, final_A].

    A "completed turn" is any turn before the most recent original
    HumanMessage — its body has been distilled into a final assistant
    answer, and the intermediate AIMessage(tool_calls=...) "thinking"
    messages and ToolMessage results are no longer load-bearing for
    future turns. Compacting them away saves 10-30x tokens on
    tool-heavy threads while preserving multi-turn coherence.

    The active turn (from the last original HumanMessage onward) is
    preserved verbatim — the ReAct loop needs to see its own in-flight
    tool calls and results.

    Turns without a final-A (interrupted / errored before producing a
    text answer) are dropped — half a turn would only confuse the model.
    """
    starts = _turn_boundaries(messages)
    if not starts:
        return list(messages)

    out: list[BaseMessage] = []
    out.extend(messages[: starts[0]])

    for i in range(len(starts) - 1):
        turn_start = starts[i]
        turn_end = starts[i + 1]
        final_idx = _final_text_ai(messages, turn_start, turn_end)
        if final_idx is None:
            continue
        out.append(messages[turn_start])  # the question
        out.append(messages[final_idx])  # the final answer

    out.extend(messages[starts[-1]:])  # active turn verbatim
    return out


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Two-stage history condensation.

    Stage 1 — Compact completed turns to [Q, final_A] pairs. Drops
    intermediate AIMessage(tool_calls=...) and ToolMessage results that
    were folded into the final answer. The active turn is preserved
    verbatim so the ReAct loop sees its own tool results.

    Stage 2 — If still over HISTORY_TOKEN_BUDGET, evict completed
    (Q, A) pairs from the head, oldest-first. Pairs are evicted whole
    (a Q without its A leaves a dangling reference).

    Stage 3 — If even with no completed pairs the active turn alone
    exceeds budget, trim within the active turn. Picks `start_on`
    dynamically: if any AIMessage is present (normal mid-ReAct state),
    `start_on="ai"` so ToolMessages stay paired with their parent;
    otherwise (post-`_convert_tool_messages_to_human` state in finalize,
    where AIMessage(tool_calls) was stripped and ToolMessages turned
    into synthesized HumanMessages), `start_on="human"` so the
    synthesized tool-results-as-Humans can be the kept window.
    Without that dynamic check, the start_on="ai" constraint would
    match nothing in the post-conversion state and finalize would lose
    every retrieved tool result.
    """
    compacted = _compact_completed_turns(messages)

    last_human = next(
        (i for i in range(len(compacted) - 1, -1, -1)
         if _is_original_user_message(compacted[i])),
        None,
    )
    if last_human is None:
        return trim_messages(
            compacted,
            max_tokens=HISTORY_TOKEN_BUDGET,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            end_on=("human", "tool"),
            allow_partial=False,
        )

    head = compacted[:last_human]
    tail = compacted[last_human:]
    tail_tokens = count_tokens_approximately(tail)

    # Stage 3: active turn alone exceeds budget. Drop head, trim within tail.
    if tail_tokens >= HISTORY_TOKEN_BUDGET:
        question = tail[0]
        rest = tail[1:]
        budget = max(0, HISTORY_TOKEN_BUDGET - count_tokens_approximately([question]))
        if budget == 0 or not rest:
            return [question]
        has_ai = any(isinstance(m, AIMessage) for m in rest)
        start_on = "ai" if has_ai else "human"
        kept_rest = trim_messages(
            rest,
            max_tokens=budget,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on=start_on,
            end_on=("human", "tool"),
            allow_partial=False,
        )
        return [question] + kept_rest

    # Stage 2: active turn fits. After compaction, head is structured
    # as [H, A, H, A, ...] from completed turns. Evict oldest pairs whole.
    budget = HISTORY_TOKEN_BUDGET - tail_tokens
    while head and count_tokens_approximately(head) > budget:
        next_h = next(
            (i for i in range(1, len(head)) if _is_original_user_message(head[i])),
            None,
        )
        head = head[next_h:] if next_h is not None else []

    return head + tail


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _cached_system(text: str) -> SystemMessage:
    """Build the agent's system message.

    On Bedrock, append a cachePoint content block so Converse caches the
    prefix across ReAct turns. On OpenAI-compatible providers (DeepSeek)
    that content-block shape is unknown, so we emit a plain SystemMessage
    and rely on the provider's own prefix caching if any.
    """
    if not orchestrator_is_bedrock():
        return SystemMessage(content=text)
    return SystemMessage(
        content=[
            {"type": "text", "text": text},
            {"cachePoint": {"type": "default"}},
        ]
    )


def with_cache_on_last(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Append a Bedrock cachePoint to the content of the last message.

    On each ReAct turn, the agent node calls the LLM with a growing list
    of messages. By marking the end of the current history as a cache
    point, Bedrock caches the prefix; the next turn reads the same
    prefix at ~10% of input-token price.

    No-op for non-Bedrock orchestrators — DeepSeek applies prefix caching
    server-side with no client-side markers required.
    """
    if not orchestrator_is_bedrock():
        return messages
    if not messages:
        return messages
    last = messages[-1]
    content = last.content
    cp_block = {"cachePoint": {"type": "default"}}
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content}, cp_block]
    elif isinstance(content, list):
        if any(isinstance(b, dict) and "cachePoint" in b for b in content):
            return messages
        new_content = list(content) + [cp_block]
    else:
        return messages
    new_last = last.model_copy(update={"content": new_content})
    return list(messages[:-1]) + [new_last]


# --- Chain factories -------------------------------------------------------

def _build_agent_system(customer_name: str) -> str:
    return (
        AGENT_SYSTEM_PROMPT
        .replace("{customer_name}", customer_name)
        .replace("{filings_catalog}", format_catalog())
    )


def get_agent_chain(customer_name: str = "Guest") -> Runnable:
    model = get_model(temperature=_agent_temp()).bind_tools(get_active_tools())
    system = _build_agent_system(customer_name)
    prompt = ChatPromptTemplate.from_messages(
        [_cached_system(system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_finalize_chain(customer_name: str = "Guest") -> Runnable:
    """Agent chain WITHOUT tools bound — used to force a text answer
    when the ReAct tool-call budget is exhausted."""
    model = get_model(temperature=_agent_temp())
    system = _build_agent_system(customer_name) + (
        "\n\nYou have already gathered research via tool calls and your tool "
        "budget is now exhausted. Do NOT attempt any more tool calls. Produce "
        "the best final answer you can from the tool results already in the "
        "conversation history. If the information is insufficient, say so "
        "clearly and explain what is missing."
    )
    prompt = ChatPromptTemplate.from_messages(
        [_cached_system(system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_router_chain() -> Runnable:
    model = get_model(temperature=0.0, router=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_simple_response_chain(customer_name: str = "Guest") -> Runnable:
    model = get_model(temperature=0.7)
    system = SIMPLE_RESPONSE_PROMPT.replace(
        "{customer_name}", _escape_braces(customer_name)
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model
