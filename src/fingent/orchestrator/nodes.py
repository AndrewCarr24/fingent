"""LangGraph nodes: router, cache check, agent (ReAct), simple response, finalize."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from fingent.models.factory import extract_text_content
from fingent.orchestrator.chains import (
    get_agent_chain,
    get_finalize_chain,
    get_router_chain,
    get_simple_response_chain,
    trim_history,
    with_cache_on_last,
)
from fingent.orchestrator.state import AgentState, IntentType


async def router_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classify intent and store it on state."""
    messages = list(state["messages"])
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": messages}, config)
    text = extract_text_content(response.content).strip().lower()

    if "rag_query" in text:
        intent: IntentType = "rag_query"
    elif "off_topic" in text:
        intent = "off_topic"
    elif "simple" in text:
        intent = "simple"
    else:
        logger.warning(f"Unclear intent: {text!r}, defaulting to rag_query")
        intent = "rag_query"

    logger.info(f"Router classified intent: {intent}")
    return {"intent": intent}


async def cache_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """Placeholder for an answer cache.

    No-op in v0.1. The hook is reserved for a future answer-cache
    implementation (e.g. exact-match or embedding-based recall of
    previous Q→A pairs within a thread). For now, the actual caching
    happens at the LLM layer via Bedrock's `cachePoint` blocks (see
    chains.py) and DeepSeek's automatic server-side prefix caching.
    """
    return {"cache_hit": False}


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Run the ReAct agent for rag_query intents."""
    messages = list(state["messages"])
    tool_call_count = state.get("tool_call_count", 0)

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    messages = trim_history(messages)
    chain = get_agent_chain(customer_name=customer_name)
    response = await chain.ainvoke(
        {"messages": with_cache_on_last(messages)}, config
    )

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    new_count = tool_call_count + (len(response.tool_calls) if has_tool_calls else 0)
    logger.debug(
        f"agent_node: has_tool_calls={has_tool_calls}, tool_call_count={new_count}"
    )
    return {"messages": response, "tool_call_count": new_count}


async def finalize_node(state: AgentState, config: RunnableConfig) -> dict:
    """Force a text answer after the ReAct tool budget is exhausted.

    Collapses the tool-call/tool-result message pairs into plain
    HumanMessages so Bedrock doesn't require a toolConfig, then asks
    the model (without tools) to synthesize a final answer.
    """
    from langchain_core.messages import ToolMessage

    raw_messages = list(state["messages"])
    condensed = []
    for msg in raw_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        if isinstance(msg, ToolMessage):
            condensed.append(HumanMessage(
                content=f"[Tool result for '{msg.name}']\n{msg.content}"
            ))
            continue
        condensed.append(msg)

    condensed = trim_history(condensed)
    logger.debug(f"finalize_node: condensed {len(raw_messages)} msgs → {len(condensed)}")

    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    chain = get_finalize_chain(customer_name=customer_name)
    response = await chain.ainvoke({"messages": condensed}, config)
    logger.info("finalize_node: produced fallback answer")
    return {"messages": response}


async def simple_response_node(state: AgentState, config: RunnableConfig) -> dict:
    """Handle greetings/thanks/off-topic without tools."""
    messages = list(state["messages"])
    configurable = config.get("configurable", {})
    customer_name = configurable.get("customer_name", "Guest")

    chain = get_simple_response_chain(customer_name=customer_name)
    response = await chain.ainvoke({"messages": messages}, config)
    return {"messages": response}
