"""LangGraph workflow: router + ReAct agent with KB tool + finalize fallback.

Topology:

    START -> router_node -> [intent?]
                              ├── rag_query  -> cache_check -> agent_node <-> tool_node
                              │                                         └── finalize -> END
                              └── simple/off -> simple_response_node ---------------> END

Caching:
- LLM prefix caching is wired in via Bedrock cachePoint blocks (see
  chains.py). DeepSeek applies server-side prefix caching automatically.
- An in-process `MemorySaver` checkpointer remembers multi-turn history
  within a single agent process. State is lost on restart.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from fingent.orchestrator.chains import get_active_tools
from fingent.orchestrator.edges import route_by_intent, should_continue
from fingent.orchestrator.nodes import (
    agent_node,
    cache_check_node,
    finalize_node,
    router_node,
    simple_response_node,
)
from fingent.orchestrator.state import AgentState

_graph_instance = None


def create_graph(force_recreate: bool = False):
    """Build the agent graph (cached singleton).

    Pass force_recreate=True to rebuild — useful when the active tool
    list changes (e.g. after `Agent(extra_tools=[...])`).
    """
    global _graph_instance
    if _graph_instance is not None and not force_recreate:
        return _graph_instance

    logger.info("Creating QA agent graph (router + ReAct + finalize)")

    builder = StateGraph(AgentState)
    builder.add_node("router_node", router_node)
    builder.add_node("cache_check_node", cache_check_node)
    builder.add_node("agent_node", agent_node)
    builder.add_node("simple_response_node", simple_response_node)
    builder.add_node("finalize_node", finalize_node)
    builder.add_node("tool_node", ToolNode(get_active_tools()))

    builder.add_edge(START, "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_by_intent,
        {"cache_check": "cache_check_node", "simple_response": "simple_response_node"},
    )
    builder.add_edge("cache_check_node", "agent_node")
    builder.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tools": "tool_node",
            "finalize": "finalize_node",
            "end": END,
        },
    )
    builder.add_edge("tool_node", "agent_node")
    builder.add_edge("finalize_node", END)
    builder.add_edge("simple_response_node", END)

    _graph_instance = builder.compile(checkpointer=MemorySaver())
    logger.info("QA agent graph compiled")
    return _graph_instance


def reset_graph() -> None:
    """Drop the cached graph. Next `create_graph()` will rebuild."""
    global _graph_instance
    _graph_instance = None
