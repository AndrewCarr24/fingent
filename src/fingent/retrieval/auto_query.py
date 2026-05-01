"""Query expansion (auto-query) — decompose a user question into a small
set of independent KB search queries via a single LLM call.

This is our own implementation of dsRAG's auto-query helper. The upstream
`dsrag.auto_query.get_search_queries` is marked legacy in dsRAG's source
and is hardcoded to Claude Sonnet 3.5 via the Anthropic API; we replicate
the small logic here so it routes through our DeepSeek client. The
`AUTO_QUERY_GUIDANCE` prompt is taken from dsRAG's published FinanceBench
eval script.
"""

from __future__ import annotations

import os
from typing import List

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel

from fingent.retrieval.kb import _configure_deepseek_as_openai


AUTO_QUERY_GUIDANCE = """
The knowledge base contains text documents (e.g. SEC filings, reports, transcripts). Keep this in mind when generating search queries. The things you search for should be things that are likely to be found in these documents.

When deciding what to search for, first consider the pieces of information that will be needed to answer the question. Then, consider what to search for to find those pieces of information. For example, if the question asks what the change in revenue was from 2019 to 2020, you would want to search for the 2019 and 2020 revenue numbers in two separate search queries, since those are the two separate pieces of information needed. You should also think about where you are most likely to find the information you're looking for. If you're looking for assets and liabilities, you may want to search for the balance sheet, for example.
""".strip()


_AUTO_QUERY_SYSTEM = """\
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}"""


class _Queries(BaseModel):
    queries: List[str]


_auto_query_client = None


def _get_auto_query_client():
    """Cached instructor client routed at DeepSeek's OpenAI-compatible API.

    Uses `deepseek-chat` (not v4-flash) because instructor's default
    Mode.TOOLS sends a forced `tool_choice` that v4-flash's default thinking
    mode rejects.
    """
    global _auto_query_client
    if _auto_query_client is not None:
        return _auto_query_client
    _configure_deepseek_as_openai()
    import instructor
    from openai import OpenAI

    oa = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("DSRAG_OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=60.0,
    )
    # Wrap so the auto-query LLM round-trip (prompt, completion, tokens)
    # appears in LangSmith as a child run when LANGSMITH_TRACING=true.
    # No-op when tracing is disabled.
    oa = wrap_openai(oa)
    _auto_query_client = instructor.from_openai(oa, mode=instructor.Mode.TOOLS)
    return _auto_query_client


@traceable(run_type="chain", name="auto_query")
def get_search_queries(
    user_input: str,
    auto_query_guidance: str = AUTO_QUERY_GUIDANCE,
    max_queries: int = 6,
) -> List[str]:
    """Decompose a user question into a small set of independent KB search
    queries via a single LLM call."""
    client = _get_auto_query_client()
    seed_env = os.environ.get("FINGENT_SEED")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=400,
        temperature=float(os.environ.get("FINGENT_AUTO_QUERY_TEMPERATURE", "0.2")),
        response_model=_Queries,
        messages=[
            {
                "role": "system",
                "content": _AUTO_QUERY_SYSTEM.format(
                    max_queries=max_queries,
                    auto_query_guidance=auto_query_guidance,
                ),
            },
            {"role": "user", "content": user_input},
        ],
        **({"seed": int(seed_env)} if seed_env is not None else {}),
    )
    return resp.queries[:max_queries]
