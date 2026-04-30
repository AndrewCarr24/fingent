"""CLI entry point: `fingent-build-kb <markdown-dir> [--kb-id ID] [--store-dir DIR] [doc_id ...]`."""

from __future__ import annotations

import argparse
import sys

from fingent.build_kb.ingest import build_kb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fingent-build-kb",
        description="Build (or extend) a FinGent knowledge base from markdown files.",
    )
    parser.add_argument(
        "parsed_dir",
        help="Path to directory containing markdown files (each *.md becomes one document).",
    )
    parser.add_argument(
        "--kb-id",
        default="fingent_kb",
        help="Identifier for the KB (default: fingent_kb).",
    )
    parser.add_argument(
        "--store-dir",
        default="./fingent_store",
        help="Where to persist the KB (default: ./fingent_store).",
    )
    parser.add_argument(
        "allowlist",
        nargs="*",
        help="Optional doc_id stems to ingest (default: all *.md under parsed_dir).",
    )
    args = parser.parse_args(argv)

    try:
        build_kb(
            parsed_dir=args.parsed_dir,
            kb_id=args.kb_id,
            store_dir=args.store_dir,
            allowlist=args.allowlist or None,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
