"""Chains for the router, the RAG agent, and the simple-response path.

Implements Bedrock prefix caching via `cachePoint` content blocks on the
system message and the last message of each ReAct turn. This is the
primary cost-saving mechanism — Bedrock caches the prefix at ~10% of
input-token price across ReAct iterations within a turn and across turns
within a session.

DeepSeek applies prefix caching server-side automatically; no client-
side markers are required.
"""

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from fingent.models.factory import get_model, orchestrator_is_bedrock
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

def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Cap the message history at HISTORY_TOKEN_BUDGET, keeping recent
    turns. `start_on="human"` ensures we never strand a `ToolMessage`
    without its preceding tool-call `AIMessage` (which would error)."""
    return trim_messages(
        messages,
        max_tokens=HISTORY_TOKEN_BUDGET,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        allow_partial=False,
    )


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
    model = get_model(temperature=0.35).bind_tools(get_active_tools())
    system = _build_agent_system(customer_name)
    prompt = ChatPromptTemplate.from_messages(
        [_cached_system(system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_finalize_chain(customer_name: str = "Guest") -> Runnable:
    """Agent chain WITHOUT tools bound — used to force a text answer
    when the ReAct tool-call budget is exhausted."""
    model = get_model(temperature=0.35)
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
