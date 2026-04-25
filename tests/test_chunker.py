"""Tests for the sliding-window character chunker."""

from __future__ import annotations

import pytest

from smart_rag.chunker import chunk
from smart_rag.loader import Document


def _doc(text: str) -> Document:
    return Document(id="abc123", source="mem://t", text=text)


def test_chunk_respects_max_chars_and_overlap():
    text = "abcdefghij" * 30  # 300 chars
    doc = _doc(text)
    out = chunk(doc, max_chars=80, overlap=20)

    # No chunk should exceed max_chars.
    assert all(len(c.text) <= 80 for c in out)

    # Step is max_chars - overlap = 60. Expect ceil((300-80)/60)+1 = 5 chunks.
    assert len(out) == 5

    # Each chunk's text must equal the slice it claims.
    for c in out:
        assert c.text == text[c.start : c.end]
        assert c.doc_id == doc.id
        assert c.source == doc.source

    # Adjacent chunks must overlap by exactly 20 chars (where they overlap).
    for prev, nxt in zip(out, out[1:]):
        if prev.end > nxt.start:
            assert prev.end - nxt.start == 20

    # Final chunk must reach the end of the document.
    assert out[-1].end == len(text)


def test_empty_document_returns_no_chunks():
    out = chunk(_doc(""), max_chars=100, overlap=10)
    assert out == []


def test_short_document_yields_single_chunk():
    text = "short doc"
    out = chunk(_doc(text), max_chars=100, overlap=10)
    assert len(out) == 1
    assert out[0].text == text
    assert out[0].start == 0
    assert out[0].end == len(text)


def test_invalid_params_raise():
    doc = _doc("hello world")
    with pytest.raises(ValueError):
        chunk(doc, max_chars=0)
    with pytest.raises(ValueError):
        chunk(doc, max_chars=10, overlap=-1)
    with pytest.raises(ValueError):
        chunk(doc, max_chars=10, overlap=10)
