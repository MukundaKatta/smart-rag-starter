"""Command-line interface for smart-rag-starter.

Subcommands:
    retrieve   Load -> chunk -> retrieve -> build prompt for a query.

The CLI is intentionally small. The same building blocks are exposed as a
Python API in `smart_rag` so users can wire their own pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .chunker import chunk
from .loader import load_documents
from .prompt import build_prompt
from .retriever import BM25Retriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srag",
        description=(
            "smart-rag-starter: load text files, chunk them, retrieve "
            "with BM25, and build a prompt for an LLM."
        ),
    )
    parser.add_argument("--version", action="version", version=f"srag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    retrieve = sub.add_parser(
        "retrieve",
        help="retrieve top-k chunks for a query and print a prompt",
    )
    retrieve.add_argument("query", help="natural-language query")
    retrieve.add_argument(
        "paths",
        nargs="+",
        help="files, directories, or globs to ingest (md/txt/py/rst)",
    )
    retrieve.add_argument("--k", type=int, default=5, help="top-k chunks to retrieve")
    retrieve.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="max characters per chunk",
    )
    retrieve.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="overlap (chars) between adjacent chunks",
    )
    retrieve.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON object with prompt + retrieved chunks",
    )
    return parser


def _run_retrieve(args: argparse.Namespace) -> int:
    docs = load_documents(args.paths)
    if not docs:
        print(f"no documents loaded from: {' '.join(args.paths)}", file=sys.stderr)
        return 2

    chunks = []
    for d in docs:
        chunks.extend(chunk(d, max_chars=args.max_chars, overlap=args.overlap))
    if not chunks:
        print("no chunks produced (documents empty?)", file=sys.stderr)
        return 2

    retriever = BM25Retriever().fit(chunks)
    hits = retriever.search(args.query, k=args.k)
    retrieved = [c for c, _ in hits]
    prompt = build_prompt(args.query, retrieved)

    if args.json:
        payload = {
            "query": args.query,
            "k": args.k,
            "prompt": prompt,
            "results": [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "start": c.start,
                    "end": c.end,
                    "score": score,
                    "text": c.text,
                }
                for c, score in hits
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(prompt)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns an exit code; also usable from `python -m`."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        return _run_retrieve(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable; argparse exits


if __name__ == "__main__":
    raise SystemExit(main())
