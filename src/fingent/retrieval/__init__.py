"""Retrieval layer.

Built on top of dsRAG (https://github.com/D-Star-AI/dsRAG) — FinGent
contributes the hybrid (BM25 + vector) layer, the smart-α weighting,
the auto-query expansion routed through DeepSeek, and the LangChain
tool wrapper. dsRAG itself provides the chunking, AutoContext, and
storage backends.
"""

from fingent.retrieval.auto_query import AUTO_QUERY_GUIDANCE, get_search_queries
from fingent.retrieval.hybrid import HybridKnowledgeBase
from fingent.retrieval.kb import get_kb, set_kb_location
from fingent.retrieval.smart_alpha import smart_rrf_alpha
from fingent.retrieval.tool import search_kb

__all__ = [
    "AUTO_QUERY_GUIDANCE",
    "HybridKnowledgeBase",
    "get_kb",
    "get_search_queries",
    "search_kb",
    "set_kb_location",
    "smart_rrf_alpha",
]
