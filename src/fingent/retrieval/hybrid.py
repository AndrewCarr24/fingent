"""Hybrid retrieval KB: vector (cosine) + BM25 (lexical) with RRF fusion.

Drop-in subclass of dsRAG's `KnowledgeBase`. Overrides `_search` to combine
two parallel rankings — semantic and lexical — via Reciprocal Rank Fusion.

Why this exists
---------------
Pure semantic retrieval (the default in dsRAG with `BasicVectorDB`) blurs
queries against a cloud of related concepts. Technical phrases like
specific dollar figures, named line items, or exact terminology carry
signal that BM25 surfaces well (rare phrase → high IDF) but cosine
distance can wash out. Hybrid gives both signals a vote.

Also fixes a quiet bug: dsRAG's `BasicVectorDB.search` accepts
`metadata_filter` as an argument but never applies it (it's only honored
by the cloud backends — Milvus, Postgres, Qdrant, etc.). That meant the
`doc_id` filter we pass through `search_kb` was a no-op on retrieval
in the original dsRAG path. This subclass applies the filter ourselves.

Build cost
----------
At KB load, we tokenize every chunk and build a BM25Okapi index in
memory. This happens inside the lazy-eager warmup, so it's hidden behind
the first user request.

Per-query cost
--------------
~50-100ms additional for BM25 scoring + RRF merge. Negligible vs the
LLM round-trips on either side.
"""

from __future__ import annotations

import os
import re

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from dsrag.database.vector.types import VectorSearchResult
from dsrag.knowledge_base import KnowledgeBase


# --- tokenization ----------------------------------------------------------

# Simple, robust tokenizer: lowercase + word characters only. Numbers like
# "1.5" split into "1", "5", but the BM25 signal we care about is
# multi-word technical phrases ("loss ratio", "consolidated statements
# of operations"), not exact numerical values. Easy to swap later.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --- metadata_filter support (dsRAG's filter dict spec) --------------------

def _matches_filter(md: dict, filt: dict | None) -> bool:
    """Apply a dsRAG-shaped MetadataFilter to a single metadata dict.

    Supports the operators dsRAG documents on its `MetadataFilter` TypedDict:
    equals, not_equals, in, not_in, greater_than, less_than,
    greater_than_equals, less_than_equals.
    """
    if filt is None:
        return True
    field, op, val = filt["field"], filt["operator"], filt["value"]
    actual = md.get(field)
    if op == "equals":
        return actual == val
    if op == "not_equals":
        return actual != val
    if op == "in":
        return actual in val
    if op == "not_in":
        return actual not in val
    if actual is None:
        return False
    if op == "greater_than":
        return actual > val
    if op == "less_than":
        return actual < val
    if op == "greater_than_equals":
        return actual >= val
    if op == "less_than_equals":
        return actual <= val
    return True  # unknown op → don't drop


# --- the hybrid KB ---------------------------------------------------------

class HybridKnowledgeBase(KnowledgeBase):
    """Vector + BM25 hybrid retrieval, drop-in for KnowledgeBase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Env-var toggles for A/B experiments:
        #   HYBRID_BM25=false   → skip BM25; pure semantic retrieval
        #   RETRIEVAL_TOP_K=N   → candidates per retriever (default 200)
        # Per-call attributes (set by the search_kb tool):
        #   _excluded_chunks    set of (doc_id, chunk_index) to omit
        #                       from candidates BEFORE RRF/RSE
        #   _rrf_alpha          BM25 weight in RRF (0..1, default 0.4)
        self._use_bm25 = os.environ.get("HYBRID_BM25", "true").lower() != "false"
        self._top_k_per_retriever = int(os.environ.get("RETRIEVAL_TOP_K", "200"))
        self._excluded_chunks: set | None = None
        # 0.4 = semantic-favored. 0.5 is unweighted standard RRF. The
        # FinanceBench-tuned static value is 0.4 (won the alpha sweep).
        self._rrf_alpha: float = 0.4
        if self._use_bm25:
            self._build_bm25_index()
        else:
            logger.info("HybridKnowledgeBase: BM25 disabled (HYBRID_BM25=false)")
            self._bm25 = None

    def _build_bm25_index(self) -> None:
        """Build a BM25 index parallel to vector_db.metadata.

        We rely on `vector_db.metadata[i]["chunk_text"]` being the raw
        chunk text. BM25 scores the same text the chunk_db.get_chunk_text
        would return.
        """
        chunks = self.vector_db.metadata
        if not chunks:
            logger.warning("HybridKnowledgeBase: vector_db has no chunks")
            self._bm25 = None
            return
        logger.info(
            f"HybridKnowledgeBase: building BM25 over {len(chunks):,} chunks"
        )
        corpus_tokens = [_tokenize(m["chunk_text"]) for m in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info("HybridKnowledgeBase: BM25 ready")

    def _filtered_indices(self, metadata_filter: dict | None) -> np.ndarray:
        """Return numpy array of indices into vectors/metadata that match
        the filter. Without a filter, returns all indices."""
        n = len(self.vector_db.metadata)
        if metadata_filter is None:
            return np.arange(n, dtype=int)
        return np.fromiter(
            (i for i, m in enumerate(self.vector_db.metadata)
             if _matches_filter(m, metadata_filter)),
            dtype=int,
        )

    def _vector_search_filtered(
        self, query_vector, top_k: int, metadata_filter: dict | None
    ) -> list[VectorSearchResult]:
        """Cosine search with filter applied. Replaces BasicVectorDB.search,
        which silently drops the filter."""
        idx = self._filtered_indices(metadata_filter)
        if len(idx) == 0:
            return []
        vectors = np.asarray(self.vector_db.vectors)
        sims = cosine_similarity([query_vector], vectors[idx])[0]
        local_top = np.argsort(-sims)[: int(top_k)]
        return [
            VectorSearchResult(
                doc_id=self.vector_db.metadata[int(idx[li])].get("doc_id"),
                vector=None,
                metadata=self.vector_db.metadata[int(idx[li])],
                similarity=float(sims[li]),
            )
            for li in local_top
        ]

    def _bm25_search_filtered(
        self, query: str, top_k: int, metadata_filter: dict | None
    ) -> list[VectorSearchResult]:
        if self._bm25 is None:
            return []
        idx = self._filtered_indices(metadata_filter)
        if len(idx) == 0:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        scores_subset = scores[idx]
        local_top = np.argsort(-scores_subset)[: int(top_k)]
        return [
            VectorSearchResult(
                doc_id=self.vector_db.metadata[int(idx[li])].get("doc_id"),
                vector=None,
                metadata=self.vector_db.metadata[int(idx[li])],
                similarity=float(scores_subset[li]),  # raw BM25 score
            )
            for li in local_top
        ]

    def _drop_excluded(self, hits: list[VectorSearchResult]) -> list[VectorSearchResult]:
        """Remove hits whose (doc_id, chunk_index) is in self._excluded_chunks."""
        if not self._excluded_chunks:
            return hits
        excl = self._excluded_chunks
        return [
            h for h in hits
            if (h["metadata"]["doc_id"], h["metadata"]["chunk_index"]) not in excl
        ]

    def _search(self, query: str, top_k: int, metadata_filter=None) -> list:
        """Hybrid retrieval: vector (+ optional BM25), optionally fused via RRF.

        Each retriever returns up to `self._top_k_per_retriever` candidates
        (env: RETRIEVAL_TOP_K, default 200). When BM25 is on, RRF merges
        the two ranked lists with weighting controlled by
        `self._rrf_alpha` (BM25 weight, 0..1; default 0.4 — semantic
        dominant with BM25 as tiebreaker; 0.5 is unweighted standard
        RRF). When `self._excluded_chunks` is set, those chunks are
        dropped from candidates before RRF/RSE — used to dedup chunks
        across multiple tool calls in one conversation thread.
        """
        N = self._top_k_per_retriever

        # Vector path (with filter applied, since BasicVectorDB ignores it)
        query_vector = self._get_embeddings([query], input_type="query")[0]
        vec_hits = self._drop_excluded(
            self._vector_search_filtered(query_vector, N, metadata_filter)
        )

        if self._use_bm25:
            bm25_hits = self._drop_excluded(
                self._bm25_search_filtered(query, N, metadata_filter)
            )

            # Weighted RRF: score(d) = α * 1/(k + rank_BM25)
            #                       + (1-α) * 1/(k + rank_vec)
            # α = 0.5 (default) is equivalent to standard unweighted RRF
            # (constant scale factor doesn't affect the ranking).
            RRF_K = 60
            alpha = self._rrf_alpha
            scores: dict[tuple[str, int], float] = {}
            first_seen: dict[tuple[str, int], VectorSearchResult] = {}

            def add(hits: list[VectorSearchResult], weight: float) -> None:
                for rank, h in enumerate(hits):
                    md = h["metadata"]
                    key = (md["doc_id"], md["chunk_index"])
                    scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
                    if key not in first_seen:
                        first_seen[key] = h

            add(bm25_hits, alpha)
            add(vec_hits, 1.0 - alpha)

            merged = sorted(scores.items(), key=lambda kv: -kv[1])[: int(top_k)]
            results = [first_seen[k] for k, _ in merged]
        else:
            # Pure semantic — no fusion, just take top-`top_k` from vector hits
            results = vec_hits[: int(top_k)]

        # Run any configured reranker (NoReranker is a no-op; preserves
        # the same hook point as dsRAG's original _search).
        return self.reranker.rerank_search_results(query, results)
