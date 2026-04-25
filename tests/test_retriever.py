"""Tests for the BM25 retriever."""

from __future__ import annotations

from smart_rag.chunker import Chunk
from smart_rag.retriever import BM25Retriever


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{idx}",
        doc_id=f"d{idx}",
        source=f"mem://doc-{idx}.txt",
        text=text,
        start=0,
        end=len(text),
    )


def _corpus() -> list[Chunk]:
    return [
        _chunk(
            1,
            "Our refund policy allows refunds within 30 days of purchase. "
            "Contact support to start a refund request.",
        ),
        _chunk(
            2,
            "Cats are small, carnivorous mammals often kept as pets. "
            "Domestic cats are valued for companionship.",
        ),
        _chunk(
            3,
            "Python is a high-level, general-purpose programming language. "
            "It emphasizes code readability.",
        ),
        _chunk(
            4,
            "Shipping usually takes 3 to 5 business days within the continental US.",
        ),
    ]


def test_refund_query_returns_refund_chunk_top1():
    chunks = _corpus()
    bm25 = BM25Retriever().fit(chunks)
    hits = bm25.search("refund policy", k=3)

    assert hits, "expected non-empty results"
    top_chunk, top_score = hits[0]
    assert top_chunk.id == "c1"
    assert top_score > 0


def test_search_before_fit_raises():
    bm25 = BM25Retriever()
    try:
        bm25.search("x")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when searching before fit")


def test_zero_k_returns_empty():
    bm25 = BM25Retriever().fit(_corpus())
    assert bm25.search("refund", k=0) == []


def test_nonmatching_query_returns_empty():
    bm25 = BM25Retriever().fit(_corpus())
    # No corpus token matches this query.
    assert bm25.search("zzqqxx", k=5) == []


def test_results_are_score_sorted_desc():
    bm25 = BM25Retriever().fit(_corpus())
    hits = bm25.search("python programming language readability", k=4)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_custom_tokenizer_is_respected():
    # A tokenizer that returns nothing makes every search empty.
    bm25 = BM25Retriever(tokenizer=lambda _t: []).fit(_corpus())
    assert bm25.search("refund", k=3) == []
