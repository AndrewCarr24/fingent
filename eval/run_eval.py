"""Run FinGent over a CSV of (question, expected_answer) pairs and grade.

Usage:
    cd fingent
    uv run python eval/run_eval.py --name smartA_run1
    RRF_ALPHA=0.4 uv run python eval/run_eval.py --name alpha04_run1

The agent's KB pointer comes from .env (FINGENT_KB_ID + FINGENT_STORE_DIR).
Temperatures are 0 throughout for low-noise evals.

Outputs:
    eval/results/<name>_<ts>.csv    per-question rows
    eval/logs.json                   append-only run log
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Pin temperatures to 0 BEFORE importing fingent — minimizes sampling
# noise so accuracy variance reflects hyperparameter changes, not model
# jitter. Production defaults (0.35 agent, 0.2 auto-query) stay in the
# code; this override is eval-scoped only.
os.environ.setdefault("FINGENT_AGENT_TEMPERATURE", "0.0")
os.environ.setdefault("FINGENT_AUTO_QUERY_TEMPERATURE", "0.0")

from langchain_aws import ChatBedrockConverse  # noqa: E402
from loguru import logger  # noqa: E402

from fingent.agent import Agent  # noqa: E402
from fingent.config import settings  # noqa: E402
from fingent.models.factory import extract_text_content  # noqa: E402
from fingent.models.usage import UsageCollector, cost_usd  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
LOG_FILE = EVAL_DIR / "logs.json"

JUDGE_SYSTEM = (
    "You grade whether an assistant's answer is substantively correct given a "
    "question and a verified expected answer. Numeric values within a 1% "
    "rounding tolerance count as correct. Extra context is fine as long as "
    "the core figures/direction match. A partially-correct answer (some "
    "values right, some wrong) is INCORRECT. Reply with ONE line: "
    "'CORRECT: <=20 word reason' or 'INCORRECT: <=20 word reason'."
)


def judge(question: str, expected: str, actual: str, collector: UsageCollector) -> tuple[bool, str]:
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
    resp = llm.invoke(
        [("system", JUDGE_SYSTEM), ("user", prompt)],
        config={"callbacks": [collector]},
    )
    text = extract_text_content(resp.content).strip()
    first_line = text.splitlines()[0] if text else ""
    verdict, _, rationale = first_line.partition(":")
    correct = verdict.strip().upper() == "CORRECT"
    return correct, rationale.strip() or first_line


def _append_log(entry: dict) -> None:
    existing = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    existing.append(entry)
    LOG_FILE.write_text(json.dumps(existing, indent=2))


async def main(csv_path: Path, name: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    kb_id = os.environ.get("FINGENT_KB_ID")
    store_dir = os.environ.get("FINGENT_STORE_DIR")
    if not kb_id or not store_dir:
        raise SystemExit(
            "FINGENT_KB_ID and FINGENT_STORE_DIR must be set (e.g. via .env)."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rrf_alpha = os.environ.get("RRF_ALPHA", "smart")
    logger.info(
        f"Eval run {name!r}: kb_id={kb_id} store_dir={store_dir} "
        f"RRF_ALPHA={rrf_alpha} HYBRID_BM25={os.environ.get('HYBRID_BM25', 'true')}"
    )

    collector = UsageCollector()
    agent = Agent(kb_id=kb_id, store_dir=store_dir, customer_name="Evaluator")

    def _total_cost() -> float:
        return sum(
            cost_usd(m, u.input_tokens, u.output_tokens,
                     u.cache_read_tokens, u.cache_creation_tokens)
            for m, u in collector.by_model.items()
        )

    results = []
    run_start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        expected = row["expected_answer"]
        logger.info(f"[{i}/{len(rows)}] {q}")
        c0 = _total_cost()
        tc0 = len(collector.tool_calls)
        t0 = time.perf_counter()
        try:
            chunks: list[str] = []
            async for chunk in agent.ask_stream(
                q, conversation_id=f"eval-{ts}-{i}", callbacks=[collector]
            ):
                chunks.append(chunk)
            actual = "".join(chunks).strip()
        except Exception as e:
            logger.error(f"agent error on row {i}: {e}")
            actual = f"[agent error: {type(e).__name__}: {e}]"
        agent_seconds = time.perf_counter() - t0
        c1 = _total_cost()
        per_q_tools = collector.tool_calls[tc0:]
        correct, rationale = judge(q, expected, actual, collector)
        c2 = _total_cost()
        total_tool_tokens = sum(t.result_tokens_est for t in per_q_tools)
        logger.info(
            f"  -> {'OK' if correct else 'MISS'} | ${c1 - c0:.4f} agent / "
            f"${c2 - c1:.4f} judge | {len(per_q_tools)} tool calls, "
            f"{total_tool_tokens:,} tool-result tokens | "
            f"{agent_seconds:.1f}s | {rationale}"
        )
        results.append({
            "question": q,
            "expected_answer": expected,
            "agent_answer": actual,
            "correct": correct,
            "rationale": rationale,
            "agent_cost_usd": round(c1 - c0, 6),
            "judge_cost_usd": round(c2 - c1, 6),
            "n_tool_calls": len(per_q_tools),
            "tool_result_tokens_est": total_tool_tokens,
            "agent_seconds": round(agent_seconds, 2),
        })
    run_seconds = time.perf_counter() - run_start

    out_csv = RESULTS_DIR / f"{name}_{ts}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)

    usage_summary = {
        model_id: {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
            "calls": u.calls,
            "cost_usd": round(
                cost_usd(model_id, u.input_tokens, u.output_tokens,
                         u.cache_read_tokens, u.cache_creation_tokens),
                6,
            ),
        }
        for model_id, u in collector.by_model.items()
    }
    total_cost = round(sum(m["cost_usd"] for m in usage_summary.values()), 6)

    _append_log({
        "name": name,
        "input_df": csv_path.name,
        "agent_ts": ts,
        "rrf_alpha": rrf_alpha,
        "hybrid_bm25": os.environ.get("HYBRID_BM25", "true"),
        "retrieval_top_k": os.environ.get("RETRIEVAL_TOP_K", "200"),
        "orchestrator_provider": settings.ORCHESTRATOR_PROVIDER,
        "orchestrator_model": (
            settings.DEEPSEEK_MODEL_ID
            if settings.ORCHESTRATOR_PROVIDER == "deepseek"
            else settings.ORCHESTRATOR_MODEL_ID
        ),
        "accuracy": round(accuracy, 4),
        "n": len(results),
        "n_correct": n_correct,
        "results_file": str(out_csv.relative_to(EVAL_DIR)),
        "usage": usage_summary,
        "total_cost_usd": total_cost,
        "run_seconds": round(run_seconds, 2),
    })

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.1%} ({n_correct}/{len(results)})")
    print(f"Time:     {run_seconds:.1f}s")
    print(f"Cost:     ${total_cost:.4f}")
    print(f"Results:  {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(EVAL_DIR / "questions_mi.csv"),
        help="Input CSV (default: eval/questions_mi.csv).",
    )
    parser.add_argument("--name", required=True, help="Run label, e.g. smartA_run1.")
    args = parser.parse_args()
    asyncio.run(main(Path(args.csv_path), args.name))
