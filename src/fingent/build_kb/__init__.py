"""Markdown → KB pipeline.

Public API:
    >>> from fingent import build_kb
    >>> build_kb(parsed_dir="./my_docs", kb_id="my_kb", store_dir="./fingent_store")

Or via CLI:
    $ fingent-build-kb ./my_docs --kb-id my_kb --store-dir ./fingent_store
"""

from fingent.build_kb.ingest import build_kb

__all__ = ["build_kb"]
