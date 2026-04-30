"""`search_kb` — the LangChain tool the agent uses to retrieve from the KB.

Built on top of dsRAG (https://github.com/D-Star-AI/dsRAG): the underlying
chunking, AutoContext, and storage backends are dsRAG's. FinGent layers
hybrid retrieval (BM25 + vector via RRF), smart-α weighting, auto-query
expansion, and per-thread chunk dedup on top.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from loguru import logger


# Per-thread set of (doc_id, chunk_index) tuples we've already returned in
# earlier calls within the same conversation. Used when DEDUP_CHUNKS=true
# to keep subsequent calls from re-pulling the same content. Keyed by
# thread_id from RunnableConfig.
_SEEN_CHUNKS_PER_THREAD: dict[str, set] = {}


@tool
def search_kb(
    question: str,
    doc_id: str | None = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """Semantic + lexical search over the knowledge base.

    Pass the user's question verbatim — the tool decomposes it into
    multiple search queries internally (auto-query helper) and runs them
    all against the KB. Returns the most relevant multi-chunk *segments*
    (contiguous sections identified by dsRAG's Relevant Segment
    Extraction). Segments include an AutoContext header describing the
    source document and section.

    Scoping: pass `doc_id` to restrict retrieval to a single document
    (recommended whenever the user's question names a specific source).
    The KB holds multiple documents; without a filter, results can come
    from any of them. Pass doc_id=None to search across all documents
    (appropriate for cross-document comparisons).

    Args:
        question: The user's question (verbatim; do not paraphrase).
        doc_id: Optional document identifier to restrict retrieval to.

    Returns:
        JSON list of {score, doc_id, content} segments, highest score
        first.
    """
    from fingent.retrieval.auto_query import get_search_queries
    from fingent.retrieval.kb import get_kb
    from fingent.retrieval.smart_alpha import smart_rrf_alpha

    try:
        queries = get_search_queries(question, max_queries=6)
    except Exception as e:
        logger.warning(f"search_kb auto-query failed: {e}")
        queries = [question]

    kb = get_kb()
    metadata_filter = (
        {"field": "doc_id", "operator": "equals", "value": doc_id}
        if doc_id
        else None
    )

    # --- env-var-driven knobs ------------------------------------------------
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "_default")

    # Chunk dedup: when DEDUP_CHUNKS=true, exclude chunks already returned
    # in earlier calls in this thread.
    dedup_on = os.environ.get("DEDUP_CHUNKS", "false").lower() == "true"
    seen = _SEEN_CHUNKS_PER_THREAD.setdefault(thread_id, set()) if dedup_on else None
    kb._excluded_chunks = seen if dedup_on else None

    # RRF alpha: BM25 weight in fusion. Default "smart" — DeepSeek picks
    # per question. Numeric value (e.g. "0.4") forces a static alpha.
    alpha_raw = os.environ.get("RRF_ALPHA", "smart").strip()
    if alpha_raw.lower() == "smart":
        alpha = smart_rrf_alpha(question)
        logger.info(f"search_kb: smart α={alpha:.2f} for question {question[:60]!r}")
    else:
        try:
            alpha = max(0.0, min(1.0, float(alpha_raw)))
        except ValueError:
            alpha = 0.5
    kb._rrf_alpha = alpha

    logger.info(
        f"search_kb invoked: question={question[:80]!r} doc_id={doc_id!r} "
        f"expanded_to={queries} α={alpha:.2f} dedup={dedup_on}"
    )

    try:
        results = kb.query(queries, metadata_filter=metadata_filter)
    except Exception as e:
        logger.warning(f"search_kb query failed: {e}")
        return json.dumps({"error": str(e)})
    finally:
        # Reset per-call attrs so a stale value can't leak into the next call.
        kb._excluded_chunks = None
        kb._rrf_alpha = 0.4

    # If dedup is on, mark every chunk that contributed to a returned
    # segment as "seen" so it's excluded from future calls in this thread.
    if dedup_on and seen is not None:
        for r in results:
            doc = r.get("doc_id", "")
            cs, ce = r.get("chunk_start"), r.get("chunk_end")
            if cs is not None and ce is not None:
                for ci in range(int(cs), int(ce) + 1):
                    seen.add((doc, ci))

    payload = [
        {
            "score": round(float(r.get("score", 0.0) or 0.0), 3),
            "doc_id": r.get("doc_id", ""),
            "content": (r.get("content") or r.get("text") or "")[:6000],
        }
        for r in results
    ]
    return json.dumps(payload, indent=2, default=str)


def get_default_tools() -> list:
    """Return the default tool set the agent ships with.

    Currently just `search_kb`. Users can pass `extra_tools` to the
    Agent class to register additional LangChain tools alongside this.
    """
    return [search_kb]
