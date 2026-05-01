"""Run one hardcoded question through fingent and grade.

Used for variance experiments. Set FINGENT_SEED, RRF_ALPHA, etc. via env.
Prints the final agent answer + correctness verdict from the Bedrock Haiku
judge (same judge as run_eval.py, T=0).
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

# Pin temperatures BEFORE importing fingent.
os.environ.setdefault("FINGENT_AGENT_TEMPERATURE", "0.0")
os.environ.setdefault("FINGENT_AUTO_QUERY_TEMPERATURE", "0.0")

from langchain_aws import ChatBedrockConverse  # noqa: E402

from fingent.agent import Agent  # noqa: E402
from fingent.config import settings  # noqa: E402
from fingent.models.factory import extract_text_content  # noqa: E402

# Q17 from the 23-question MI eval — exactly 50/50 split in our last run.
QUESTION = (
    "At year-end 2024, what was the dispersion in PMIERs sufficiency ratios "
    "across the six MI cohort issuers — which had the highest, which had the "
    "lowest, and what was the spread?"
)

EXPECTED = (
    "PMIERs sufficiency ratios at YE 2024 ranged from approximately 156% "
    "(Radian Guaranty, lowest, with a $2.2B/56% cushion over MRA) to 186% "
    "(Arch's combined eligible mortgage insurers AMIC and UGRIC, highest), "
    "a spread of about 30 percentage points. Other issuers: Essent 178%, "
    "NMI ~170%, Enact 167%, MGIC ~161%."
)

JUDGE_SYSTEM = (
    "You grade whether an assistant's answer is substantively correct given a "
    "question and a verified expected answer. Numeric values within a 1% "
    "rounding tolerance count as correct. Extra context is fine as long as "
    "the core figures/direction match. A partially-correct answer (some "
    "values right, some wrong) is INCORRECT. Reply with ONE line: "
    "'CORRECT: <=20 word reason' or 'INCORRECT: <=20 word reason'."
)


def judge(question: str, expected: str, actual: str) -> tuple[bool, str]:
    llm = ChatBedrockConverse(
        model=settings.ROUTER_MODEL_ID,
        region_name=settings.AWS_REGION,
        temperature=0,
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Expected answer:\n{expected}\n\n"
        f"Agent answer:\n{actual}"
    )
    resp = llm.invoke([("system", JUDGE_SYSTEM), ("user", prompt)])
    text = extract_text_content(resp.content).strip()
    first_line = text.splitlines()[0] if text else ""
    verdict, _, rationale = first_line.partition(":")
    correct = verdict.strip().upper() == "CORRECT"
    return correct, rationale.strip() or first_line


async def main() -> None:
    agent = Agent(
        kb_id=os.environ["FINGENT_KB_ID"],
        store_dir=os.environ["FINGENT_STORE_DIR"],
        customer_name="Evaluator",
    )
    t0 = time.perf_counter()
    chunks: list[str] = []
    async for c in agent.ask_stream(QUESTION, conversation_id=f"single-{int(t0)}"):
        chunks.append(c)
    answer = "".join(chunks).strip()
    elapsed = time.perf_counter() - t0
    correct, rationale = judge(QUESTION, EXPECTED, answer)

    rrf = os.environ.get("RRF_ALPHA", "smart")
    seed = os.environ.get("FINGENT_SEED", "(none)")
    label = os.environ.get("RUN_LABEL", "")
    print("=" * 72)
    print(f"RUN_LABEL={label}  RRF_ALPHA={rrf}  SEED={seed}  T_AGENT=0  T_AQ=0")
    print(f"TIME: {elapsed:.1f}s  VERDICT: {'CORRECT' if correct else 'INCORRECT'} — {rationale}")
    print("-" * 72)
    print(answer)
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
