"""Chainlit chat handler — pipes user messages into the FinGent agent.

Run:
    export FINGENT_KB_ID=my_kb
    export FINGENT_STORE_DIR=./fingent_store
    chainlit run apps/chainlit_app.py

While the agent works, intermediate reasoning + tool-call summaries
are surfaced into a collapsible cl.Step ("Working..."). The final
answer streams into the main message. Tool *results* (raw chunks) are
not shown to the user; their doc_ids are appended at the end as a
plain "Source:" line.
"""

from __future__ import annotations

import os

import chainlit as cl

from fingent.agent import Agent


# --- Agent (singleton; built lazily on first message) ----------------------
_agent: Agent | None = None


def _get_agent() -> Agent:
    global _agent
    if _agent is not None:
        return _agent
    kb_id = os.environ.get("FINGENT_KB_ID", "fingent_kb")
    store_dir = os.environ.get("FINGENT_STORE_DIR", "./fingent_store")
    _agent = Agent(kb_id=kb_id, store_dir=store_dir)
    return _agent


# --- Chainlit handlers -----------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Hi! I'm a research assistant. Ask me about any of the "
            "documents in the knowledge base."
        )
    ).send()


def _format_tool_call(tool: str, args: dict) -> str:
    if tool == "search_kb":
        question = (args.get("question") or "").strip()
        doc_id = args.get("doc_id")
        if doc_id:
            return f"🔍 Searching {doc_id} for: {question!r}"
        return f"🔍 Searching all documents for: {question!r}"
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "(no args)"
    return f"🔧 {tool}({arg_str})"


@cl.on_message
async def on_message(message: cl.Message):
    """Stream the agent's response using Chainlit's recommended Step pattern.

    The step appears as a labeled box that accumulates tool-call
    summaries and any reasoning text while the agent works. The answer
    message is created lazily on the first answer token so we never show
    an empty bubble.
    """
    agent = _get_agent()
    session_id = cl.context.session.id
    answer: cl.Message | None = None

    source_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()

    async with cl.Step(
        name="Knowledge Base",
        default_open=True,
        show_input=False,
        icon="search",
    ) as step:
        try:
            async for event in agent.ask_events(
                question=message.content,
                conversation_id=session_id,
            ):
                kind = event.get("kind")

                if kind == "answer_token":
                    if answer is None:
                        answer = cl.Message(content="")
                        await answer.send()
                    await answer.stream_token(event["text"])

                elif kind == "rewind_to_thinking":
                    rewound = event["text"]
                    if answer and answer.content.endswith(rewound):
                        answer.content = answer.content[: -len(rewound)]
                        await answer.update()
                    step.output = (step.output or "") + rewound + "\n\n"
                    await step.update()

                elif kind == "tool_call":
                    summary = _format_tool_call(event["tool"], event.get("args", {}))
                    step.output = (step.output or "") + summary + "\n\n"
                    await step.update()

                elif kind == "tool_result_segment":
                    doc_id = event.get("doc_id", "")
                    if doc_id and doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        source_doc_ids.append(doc_id)

        except Exception as e:
            if answer is None:
                answer = cl.Message(content="")
                await answer.send()
            await answer.stream_token(f"\n\n[error: {type(e).__name__}: {e}]")

    if answer is not None and source_doc_ids:
        answer.content = (answer.content or "") + "\n\nSource: " + ", ".join(source_doc_ids)

    if answer is not None:
        await answer.update()
