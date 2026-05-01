"""Interactive CLI for FinGent.

Usage:
    fingent-chat --kb-id my_kb --store-dir ./fingent_store \\
        "What did the company say about Q3 revenue?"

    # Or interactive REPL (omit the question):
    fingent-chat --kb-id my_kb --store-dir ./fingent_store

Environment:
    A `.env` file in the current working directory is auto-loaded on
    startup. `--kb-id` and `--store-dir` fall back to `FINGENT_KB_ID`
    and `FINGENT_STORE_DIR` so common workflows don't need flags.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv

# Load .env before importing Agent so the orchestrator picks up
# DEEPSEEK_API_KEY, AWS creds, retrieval knobs, etc.
load_dotenv()

from fingent.agent import Agent  # noqa: E402


async def _run_one_shot(agent: Agent, question: str, conversation_id: str | None = None) -> int:
    async for chunk in agent.ask_stream(question, conversation_id=conversation_id):
        print(chunk, end="", flush=True)
    print()
    return 0


async def _run_repl(agent: Agent) -> int:
    conversation_id = str(uuid.uuid4())
    print("FinGent REPL. Type 'exit' or Ctrl-D to quit.", file=sys.stderr)
    print(f"  conversation_id={conversation_id}", file=sys.stderr)
    print("", file=sys.stderr)

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            return 0
        async for chunk in agent.ask_stream(question, conversation_id=conversation_id):
            print(chunk, end="", flush=True)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fingent-chat",
        description="Ask questions against a FinGent knowledge base.",
    )
    parser.add_argument(
        "--kb-id",
        default=os.environ.get("FINGENT_KB_ID", "fingent_kb"),
        help="KB identifier (default: $FINGENT_KB_ID, then 'fingent_kb').",
    )
    parser.add_argument(
        "--store-dir",
        default=os.environ.get("FINGENT_STORE_DIR", "./fingent_store"),
        help="Path to KB store (default: $FINGENT_STORE_DIR, then './fingent_store').",
    )
    parser.add_argument(
        "--customer-name",
        default="User",
        help='Display name used in the system prompt (default: "User").',
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. If omitted, drops into interactive REPL.",
    )
    args = parser.parse_args(argv)

    try:
        agent = Agent(
            kb_id=args.kb_id,
            store_dir=args.store_dir,
            customer_name=args.customer_name,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.question:
        return asyncio.run(_run_one_shot(agent, args.question))
    return asyncio.run(_run_repl(agent))


if __name__ == "__main__":
    raise SystemExit(main())
