"""Build / extend a dsRAG KnowledgeBase from a folder of markdown files.

Each markdown file is ingested as its own dsRAG document — the filename
stem becomes the `doc_id`, which the agent later passes as a metadata
filter to scope retrieval to a single source.

Stack:
    Embedding:       BedrockTitanEmbedding (Titan v2 via Bedrock)
    Reranker:        NoReranker (RSE alone handles relevance scoring;
                     no external rerank API needed)
    AutoContext LLM: dsRAG's OpenAIChatAPI pointed at DeepSeek's
                     OpenAI-compatible endpoint with model="deepseek-chat"
                     (non-reasoning backend; v4-flash's default thinking
                     mode rejects instructor's forced tool_choice)
    VectorDB:        dsRAG's BasicVectorDB (file-pickle-backed)

Re-runs are idempotent: documents whose doc_id is already in the KB are
skipped (dsRAG's chunk DB tracks doc_ids already ingested).

Requires DEEPSEEK_API_KEY + AWS credentials with Bedrock access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from fingent.build_kb.bedrock_embedding import BedrockTitanEmbedding


def _configure_deepseek_as_openai() -> None:
    """Set the env vars dsRAG's OpenAIChatAPI reads, pointing at DeepSeek."""
    from fingent.config import settings

    api_key = os.environ.get("DEEPSEEK_API_KEY") or settings.DEEPSEEK_API_KEY
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.", file=sys.stderr)
        sys.exit(2)
    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["DSRAG_OPENAI_BASE_URL"] = os.environ.get(
        "DEEPSEEK_BASE_URL", settings.DEEPSEEK_BASE_URL
    )


def _already_indexed_doc_ids(kb) -> set[str]:
    try:
        return set(kb.chunk_db.get_all_doc_ids())
    except Exception:
        return set()


def build_kb(
    parsed_dir: str | Path,
    kb_id: str,
    store_dir: str | Path,
    allowlist: list[str] | None = None,
) -> None:
    """Build (or extend) a dsRAG KB from markdown under `parsed_dir`.

    Args:
        parsed_dir: Path to directory containing markdown files. Each
            `*.md` becomes one dsRAG document with `doc_id` = filename
            stem.
        kb_id: Identifier for this KB (string). Becomes the directory
            name dsRAG uses inside `store_dir`.
        store_dir: Where to persist the KB. Will be created if missing.
        allowlist: Optional list of doc_id stems to ingest. When set,
            only matching files are processed; others are skipped.

    Idempotent: documents already in the KB (by doc_id) are skipped.
    """
    _configure_deepseek_as_openai()

    from dsrag.knowledge_base import KnowledgeBase
    from dsrag.llm import OpenAIChatAPI
    from dsrag.reranker import NoReranker

    parsed_dir = Path(parsed_dir).resolve()
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    if not parsed_dir.exists():
        raise SystemExit(f"parsed_dir does not exist: {parsed_dir}")

    kb = KnowledgeBase(
        kb_id=kb_id,
        embedding_model=BedrockTitanEmbedding(),
        reranker=NoReranker(),
        auto_context_model=OpenAIChatAPI(
            model="deepseek-chat",
            temperature=0.2,
            max_tokens=2000,
        ),
        storage_directory=str(store_dir),
        exists_ok=True,
    )

    indexed = _already_indexed_doc_ids(kb)
    all_md = sorted(parsed_dir.glob("*.md"))
    if allowlist:
        wanted = set(allowlist)
        md_files = [m for m in all_md if m.stem in wanted]
        missing = wanted - {m.stem for m in md_files}
        if missing:
            raise SystemExit(
                f"No parsed markdown for doc_id(s): {sorted(missing)}. "
                f"Expected {parsed_dir}/<doc_id>.md"
            )
    else:
        md_files = all_md
    if not md_files:
        raise SystemExit(f"No markdown files to ingest under {parsed_dir}")

    logger.info(
        f"Building KB id={kb_id!r} at {store_dir}\n"
        f"  to ingest: {len(md_files)} markdown file(s)\n"
        f"  already indexed: {len(indexed)} ({sorted(indexed)})"
    )

    for md in md_files:
        doc_id = md.stem
        if doc_id in indexed:
            logger.info(f"  skip (already indexed): {doc_id}")
            continue
        text = md.read_text()
        logger.info(f"  ingesting: {doc_id} ({len(text):,} chars)")
        kb.add_document(
            doc_id=doc_id,
            text=text,
            semantic_sectioning_config={
                "use_semantic_sectioning": True,
                "llm_provider": "openai",
                "model": "deepseek-chat",
            },
            auto_context_config={
                "use_generated_title": True,
                "get_document_summary": True,
                "get_section_summaries": True,
            },
        )
    logger.info(f"Done. KB now holds: {sorted(_already_indexed_doc_ids(kb))}")
