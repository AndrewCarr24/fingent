"""Smart RRF α — DeepSeek picks the BM25 weight per question.

Used when env `RRF_ALPHA=smart` (the default). Falls back to 0.4 on any
error — that's the static value that won the FinanceBench sweep, so it's
a sensible prior if the per-question call fails.
"""

from __future__ import annotations

import os

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger

from fingent.retrieval.kb import _configure_deepseek_as_openai


_smart_alpha_client = None


def _get_smart_alpha_client():
    """Cached raw OpenAI client (no instructor) for the smart-alpha call.
    Uses the same DeepSeek endpoint as auto_query."""
    global _smart_alpha_client
    if _smart_alpha_client is not None:
        return _smart_alpha_client
    _configure_deepseek_as_openai()
    from openai import OpenAI

    oa = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("DSRAG_OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=30.0,
    )
    _smart_alpha_client = wrap_openai(oa)
    return _smart_alpha_client


_SMART_ALPHA_SYSTEM = """\
You are choosing how to weight retrieval over a knowledge base.
Two retrievers are run in parallel and their rankings are fused via RRF:
  - BM25 (lexical): rewards exact phrase matches, rare technical terms,
    specific dollar figures and percentages, exact table headings.
  - Semantic embedding: rewards conceptual overlap, paraphrase, abstract
    topic match.

Given the user's question, return a SINGLE FLOAT between 0.0 and 1.0
representing the BM25 weight in the RRF fusion. The semantic weight is
1 - alpha. Guidelines:
  - alpha=0.5: balanced (default for general questions).
  - alpha=0.65-0.75: question contains specific industry terms, named
    line items, exact metric phrases, or numerical values.
  - alpha=0.25-0.4: conceptual or abstract question ("what's the
    company's strategy", "describe risk factors") where exact phrasing
    matters less than topic overlap.

Return ONLY the float. No explanation."""


@traceable(run_type="chain", name="smart_alpha")
def smart_rrf_alpha(query: str) -> float:
    """Ask DeepSeek for the optimal BM25 weight (0-1) for this query.

    Used when env RRF_ALPHA=smart (the default). Falls back to 0.4 on
    any error.
    """
    try:
        from fingent.config import settings

        client = _get_smart_alpha_client()
        # deepseek-v4-flash is a reasoning model — it spends tokens on
        # hidden reasoning_content before emitting visible content. Cap
        # at 256 so the visible answer (a single float) actually fits.
        resp = client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL_ID,
            max_tokens=256,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SMART_ALPHA_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        alpha = float(text)
        return max(0.0, min(1.0, alpha))
    except Exception as e:
        logger.warning(f"smart_rrf_alpha failed ({e}); defaulting to 0.4")
        return 0.4
