"""Lazy singleton access to the dsRAG KnowledgeBase.

The KB location (kb_id + storage directory) is set at agent-init time
via `set_kb_location()` — the public Agent class does this for you.

Sets up environment variables so dsRAG's OpenAI client (used at query
time for AutoContext / auto-query) routes to DeepSeek's OpenAI-compatible
endpoint, matching the rest of the FinGent stack.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from loguru import logger


# --- KB location (set by the Agent class or set_kb_location) ---------------

_KB_ID: str | None = None
_KB_STORE_DIR: Path | None = None
_kb = None  # cached HybridKnowledgeBase singleton


def set_kb_location(kb_id: str, store_dir: str | Path) -> None:
    """Set the KB to load on next `get_kb()` call.

    Resets the cached singleton, so the next `get_kb()` will reload from
    the new location.
    """
    global _KB_ID, _KB_STORE_DIR, _kb
    _KB_ID = kb_id
    _KB_STORE_DIR = Path(store_dir).resolve()
    _kb = None
    logger.info(f"KB location set: kb_id={kb_id!r}, store_dir={_KB_STORE_DIR}")


def get_kb_location() -> tuple[str, Path]:
    """Return (kb_id, store_dir). Raises if not set."""
    if _KB_ID is None or _KB_STORE_DIR is None:
        raise RuntimeError(
            "KB location not set. Call set_kb_location(kb_id, store_dir) "
            "first, or use the Agent class which does this for you."
        )
    return _KB_ID, _KB_STORE_DIR


# --- DeepSeek-as-OpenAI shim ----------------------------------------------

def _configure_deepseek_as_openai() -> None:
    """dsRAG's AutoContext / auto-query LLMs route through its OpenAI client;
    point that client at DeepSeek's OpenAI-compatible endpoint.

    Pydantic Settings reads `.env` into the in-memory settings object but
    does not re-export back to os.environ. The raw OpenAI client used by
    smart_rrf_alpha and auto_query reads from os.environ directly, so we
    prime os.environ from settings here to keep both paths consistent.
    """
    from fingent.config import settings

    api_key = os.environ.get("DEEPSEEK_API_KEY") or settings.DEEPSEEK_API_KEY
    if not api_key:
        return
    os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
    # Don't stomp a real OPENAI_API_KEY if the caller set one explicitly.
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault(
        "DSRAG_OPENAI_BASE_URL",
        os.environ.get("DEEPSEEK_BASE_URL", settings.DEEPSEEK_BASE_URL),
    )


def _ensure_bedrock_embedding_registered() -> None:
    """Importing the build_kb embedding adapter registers the
    BedrockTitanEmbedding subclass with dsRAG's class registry — needed
    for `KnowledgeBase.from_dict` deserialization at load time."""
    from fingent.build_kb import bedrock_embedding  # noqa: F401


def _rewrite_kb_paths_if_needed(store_dir: Path, kb_id: str) -> None:
    """dsRAG bakes absolute paths for chunk_db / vector_db / file_system
    into the metadata JSON at build time. When the KB folder moves
    (container, fork, fresh checkout, install location), those paths are
    stale and BasicChunkDB.__init__ tries to mkdir at the old location.

    Patch the metadata in-place so paths track the current store_dir.
    Idempotent.
    """
    meta_path = store_dir / "metadata" / f"{kb_id}.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    components = meta.get("components", {})
    target_dir = str(store_dir)
    target_images = str(store_dir / "page_images")
    changed = False
    for key in ("chunk_db", "vector_db"):
        if components.get(key, {}).get("storage_directory") != target_dir:
            components.setdefault(key, {})["storage_directory"] = target_dir
            changed = True
    fs = components.get("file_system", {})
    if fs.get("base_path") != target_images:
        fs["base_path"] = target_images
        components["file_system"] = fs
        changed = True
    if changed:
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Patched KB metadata paths → {target_dir}")


# --- KB load (cached) ------------------------------------------------------

def get_kb():
    """Load the persisted KB wrapped in HybridKnowledgeBase.

    HybridKnowledgeBase subclasses dsRAG's KnowledgeBase and overrides
    `_search` to combine semantic (cosine) and lexical (BM25) retrieval
    via RRF fusion. It also applies the metadata_filter ourselves —
    BasicVectorDB silently drops it.
    """
    global _kb
    if _kb is not None:
        return _kb

    kb_id, store_dir = get_kb_location()

    _ensure_bedrock_embedding_registered()
    _configure_deepseek_as_openai()
    _rewrite_kb_paths_if_needed(store_dir, kb_id)

    from fingent.retrieval.hybrid import HybridKnowledgeBase

    logger.info(f"Loading hybrid KB '{kb_id}' from {store_dir}")
    _kb = HybridKnowledgeBase(
        kb_id,
        storage_directory=str(store_dir),
        exists_ok=True,
    )
    return _kb


def reset_kb() -> None:
    """Drop the cached KB. Next `get_kb()` will reload."""
    global _kb
    _kb = None
