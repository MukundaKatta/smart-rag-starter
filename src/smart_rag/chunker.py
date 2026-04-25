"""Sliding-window character chunker.

The chunker walks the document with a fixed-size window that advances by
`max_chars - overlap` each step. Adjacent chunks share `overlap` characters so
that context is preserved when retrieval lands near a boundary.

Future work (noted in the README roadmap): split on paragraph/sentence
boundaries first and rejoin to respect `max_chars`. The current implementation
is intentionally simple and dependency-free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    """A contiguous slice of a Document's text."""

    id: str
    doc_id: str
    source: str
    text: str
    start: int
    end: int


def _chunk_id(doc_id: str, start: int, end: int) -> str:
    """Stable id derived from doc + offsets so ids survive re-chunking."""
    raw = f"{doc_id}:{start}:{end}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def chunk(doc: Document, *, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split `doc.text` into overlapping fixed-size character windows.

    Args:
        doc: The Document to chunk.
        max_chars: Maximum characters per chunk. Must be > 0.
        overlap: How many trailing characters of one chunk are repeated at the
            start of the next. Must be >= 0 and < max_chars.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text = doc.text
    if not text:
        return []

    step = max_chars - overlap
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        slice_text = text[start:end]
        chunks.append(
            Chunk(
                id=_chunk_id(doc.id, start, end),
                doc_id=doc.id,
                source=doc.source,
                text=slice_text,
                start=start,
                end=end,
            )
        )
        if end == n:
            break
        start += step
    return chunks
