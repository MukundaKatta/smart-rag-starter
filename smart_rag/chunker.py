"""Sentence-aware chunking with configurable overlap.

The chunker splits a document into roughly equal-sized chunks, never breaking
mid-sentence when a sentence boundary is available. Adjacent chunks share a
configurable amount of overlap so context is preserved across boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .document import Document


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass
class Chunk:
    """A chunk produced by the chunker."""

    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int

    def __len__(self) -> int:
        return len(self.text)


def _split_sentences(text: str) -> List[str]:
    pieces = _SENTENCE_BOUNDARY.split(text)
    # Fall back to newline splits when there are no obvious sentence ends.
    if len(pieces) == 1:
        pieces = [p for p in text.splitlines() if p.strip()]
    return [p.strip() for p in pieces if p.strip()]


class Chunker:
    """Split documents into overlapping, sentence-aware chunks.

    `chunk_size` and `chunk_overlap` are measured in characters. Defaults are
    sensible for short notes; tune up for long-form documents.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Chunk]:
        """Split a single `Document` into a list of `Chunk` objects."""
        text = document.text.strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_len = 0
        chunk_index = 0
        char_cursor = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for joining space
            if buffer and buffer_len + sentence_len > self.chunk_size:
                chunk_text = " ".join(buffer).strip()
                start = char_cursor
                end = start + len(chunk_text)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=document.source,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1
                # Carry tail of the previous chunk for overlap.
                tail_text, tail_used = self._tail(buffer, self.chunk_overlap)
                buffer = list(tail_text)
                buffer_len = sum(len(s) + 1 for s in buffer)
                char_cursor = end - tail_used
            buffer.append(sentence)
            buffer_len += sentence_len

        if buffer:
            chunk_text = " ".join(buffer).strip()
            start = char_cursor
            end = start + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=document.source,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )

        return chunks

    def split_all(self, documents: Iterable[Document]) -> List[Chunk]:
        """Split many documents in one pass, preserving order."""
        out: List[Chunk] = []
        for doc in documents:
            out.extend(self.split(doc))
        return out

    @staticmethod
    def _tail(sentences: List[str], target_chars: int) -> "tuple[List[str], int]":
        """Return the trailing sentences whose total length is near `target_chars`."""
        if target_chars == 0 or not sentences:
            return [], 0
        tail: List[str] = []
        size = 0
        for sentence in reversed(sentences):
            candidate_size = size + len(sentence) + 1
            if candidate_size > target_chars and tail:
                break
            tail.insert(0, sentence)
            size = candidate_size
        return tail, size
