"""Public Agent API.

Usage:
    >>> from fingent import Agent
    >>> agent = Agent(kb_id="my_kb", store_dir="./fingent_store")
    >>> answer = await agent.ask("What did the company say about Q3 revenue?")
    >>> async for chunk in agent.ask_stream("..."):
    ...     print(chunk, end="", flush=True)

    # Pass extra tools alongside the default search_kb:
    >>> from fingent.retrieval.tool import search_kb
    >>> agent = Agent(
    ...     kb_id="my_kb",
    ...     store_dir="./fingent_store",
    ...     extra_tools=[my_custom_tool],
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from fingent.orchestrator.chains import set_active_tools
from fingent.orchestrator.graph import reset_graph
from fingent.orchestrator.streaming import (
    get_streaming_events,
    get_streaming_response,
)
from fingent.retrieval.kb import set_kb_location
from fingent.retrieval.tool import get_default_tools


class Agent:
    """A FinGent agent bound to one KB.

    The constructor wires up the KB location and tool set. After
    construction the agent is reusable across many `ask` / `ask_stream`
    calls. Multi-turn conversation history within a single Python
    process is preserved via LangGraph's in-process MemorySaver,
    keyed on `conversation_id` (= LangGraph thread_id).
    """

    def __init__(
        self,
        kb_id: str,
        store_dir: str | Path,
        extra_tools: list | None = None,
        customer_name: str = "User",
    ) -> None:
        """Construct the agent.

        Args:
            kb_id: Identifier of the KB (matches what you passed to
                `build_kb`).
            store_dir: Path to the KB store directory.
            extra_tools: Optional list of additional LangChain tools to
                bind to the agent. The default `search_kb` is always
                included; pass extras here to register your own tools.
            customer_name: Display name used in the system prompt
                ("You are a research assistant helping {customer_name}").
        """
        self.kb_id = kb_id
        self.store_dir = Path(store_dir).resolve()
        self.customer_name = customer_name

        if not self.store_dir.exists():
            raise FileNotFoundError(
                f"KB store directory not found: {self.store_dir}. "
                f"Run `fingent-build-kb` to create it first."
            )

        # Wire up the singleton KB location so the search_kb tool knows
        # where to load from. Setting this resets the cached KB.
        set_kb_location(kb_id=kb_id, store_dir=self.store_dir)

        # Set the active tool list. Default = [search_kb].
        tools = list(get_default_tools())
        if extra_tools:
            tools.extend(extra_tools)
        set_active_tools(tools)

        # Force a graph rebuild so the new tool list is bound.
        reset_graph()

        logger.info(
            f"Agent initialized: kb_id={kb_id!r}, store_dir={self.store_dir}, "
            f"tools={[getattr(t, 'name', repr(t)) for t in tools]}"
        )

    async def ask(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> str:
        """Ask a question; return the full final answer as one string.

        For token-streaming, use `ask_stream`.
        """
        chunks: list[str] = []
        async for c in self.ask_stream(question, conversation_id=conversation_id):
            chunks.append(c)
        return "".join(chunks)

    async def ask_stream(
        self,
        question: str,
        conversation_id: str | None = None,
        callbacks: list | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's answer token-by-token (final-answer text only).

        For tagged events (tool calls, tool result segments) suitable for
        a Chainlit-style UI, use `ask_events` instead.
        """
        async for chunk in get_streaming_response(
            messages=question,
            customer_name=self.customer_name,
            conversation_id=conversation_id,
            callbacks=callbacks,
        ):
            yield chunk

    async def ask_events(
        self,
        question: str,
        conversation_id: str | None = None,
        callbacks: list | None = None,
    ) -> AsyncIterator[dict]:
        """Stream tagged events suitable for a UI step view.

        Yields dicts with `kind` ∈ {answer_token, rewind_to_thinking,
        tool_call, tool_result_segment}. See
        `fingent.orchestrator.streaming.get_streaming_events` for the
        exact schema.
        """
        async for event in get_streaming_events(
            messages=question,
            customer_name=self.customer_name,
            conversation_id=conversation_id,
            callbacks=callbacks,
        ):
            yield event
