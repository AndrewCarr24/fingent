"""Catalog of available documents in the KB.

Reads the dsRAG `chunk_db` for the loaded KB and lists one row per
document. Used to inject a small "what's in the KB" table into the
agent's system prompt so the agent can map (entity, period) → doc_id
without calling a tool.

Generic: works for any markdown corpus. There's no domain-specific
filename parsing — doc_id is whatever the user named the markdown file
(stem) when they ran `fingent-build-kb`.
"""

from __future__ import annotations

from loguru import logger


def list_doc_ids() -> list[str]:
    """Return the doc_ids currently in the KB."""
    try:
        from fingent.retrieval.kb import get_kb

        kb = get_kb()
        ids = list(kb.chunk_db.get_all_doc_ids())
        return sorted(ids)
    except Exception as e:
        logger.warning(f"list_doc_ids failed: {e}")
        return []


def format_for_prompt() -> str:
    """Render the catalog as a one-doc_id-per-line list for the agent's
    system prompt.

    Kept intentionally minimal — FinGent doesn't know the structure of
    the user's doc_ids (they could be ticker_form_period, paper IDs,
    URLs, or anything). Just lists what's available so the agent can
    reference it as a metadata filter.
    """
    ids = list_doc_ids()
    if not ids:
        return "No documents indexed yet."
    lines = ["doc_id (filter on this)"]
    lines.extend(ids)
    return "\n".join(lines)
