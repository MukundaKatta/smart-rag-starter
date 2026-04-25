"""Tests for prompt assembly."""

from __future__ import annotations

from smart_rag.chunker import Chunk
from smart_rag.prompt import build_messages, build_prompt


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{idx}",
        doc_id=f"d{idx}",
        source=f"docs/doc-{idx}.md",
        text=text,
        start=0,
        end=len(text),
    )


def test_build_prompt_includes_query_chunks_citations_sources():
    query = "What is the refund policy?"
    chunks = [
        _chunk(1, "Refunds within 30 days of purchase."),
        _chunk(2, "Contact support to start a refund."),
    ]
    out = build_prompt(query, chunks)

    # Query is present.
    assert query in out

    # Every chunk's text appears verbatim.
    for ch in chunks:
        assert ch.text in out

    # Numbered inline citations [1] and [2] both present.
    assert "[1]" in out
    assert "[2]" in out

    # A "Sources" section listing each chunk's source.
    assert "Sources:" in out
    for ch in chunks:
        assert ch.source in out


def test_build_prompt_handles_empty_retrieval():
    out = build_prompt("anything?", [])
    assert "anything?" in out
    assert "Sources:" in out
    assert "(no context retrieved)" in out


def test_build_messages_returns_role_pairs():
    chunks = [_chunk(1, "Hello world.")]
    msgs = build_messages("hi?", chunks, system="be terse")
    assert isinstance(msgs, list)
    assert {m["role"] for m in msgs} == {"system", "user"}
    user = next(m for m in msgs if m["role"] == "user")
    assert "Hello world." in user["content"]
    assert "[1]" in user["content"]
    assert "hi?" in user["content"]
    sys_msg = next(m for m in msgs if m["role"] == "system")
    assert sys_msg["content"] == "be terse"
