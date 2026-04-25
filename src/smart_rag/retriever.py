"""Pure-Python BM25 retriever over Chunk objects.

Implements Robertson/Zaragoza BM25 with the standard parameters:

    score(q, d) = sum over terms t in q of:
        IDF(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

with the "+0.5" IDF smoothing that keeps scores positive even on small
corpora:

    IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

The tokenizer is a simple lowercase regex split on `\\w+`. It can be
overridden via the constructor for callers that need stemming, stopwords, or
language-specific behavior.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable

from .chunker import Chunk


_DEFAULT_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def default_tokenizer(text: str) -> list[str]:
    """Lowercase, regex-based tokenizer used when no custom one is supplied."""
    return _DEFAULT_TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 ranker. Call `.fit(chunks)` then `.search(query, k)`."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or default_tokenizer

        # Index state. Populated by `.fit`.
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        # Map term -> document frequency (number of chunks containing the term).
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, chunks: Iterable[Chunk]) -> "BM25Retriever":
        """Build the index from `chunks`. Returns `self` for chaining."""
        self._chunks = list(chunks)
        self._term_freqs = []
        self._doc_lens = []
        self._df = {}

        for ch in self._chunks:
            tokens = self.tokenizer(ch.text)
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        n = len(self._chunks)
        total_len = sum(self._doc_lens)
        self._avgdl = (total_len / n) if n else 0.0

        # Precompute IDF with the standard "+0.5 / +1" smoothing.
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self._df.items()
        }
        self._fitted = True
        return self

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        if self._avgdl == 0:
            return 0.0
        tf = self._term_freqs[doc_index]
        dl = self._doc_lens[doc_index]
        norm = 1 - self.b + self.b * (dl / self._avgdl)
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * norm
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return up to `k` (Chunk, score) pairs ranked by BM25 score, desc."""
        if not self._fitted:
            raise RuntimeError("BM25Retriever.search called before .fit()")
        if k <= 0:
            return []
        query_terms = self.tokenizer(query)
        if not query_terms:
            return []

        scored: list[tuple[Chunk, float]] = []
        for i, ch in enumerate(self._chunks):
            s = self._score(query_terms, i)
            if s > 0:
                scored.append((ch, s))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
