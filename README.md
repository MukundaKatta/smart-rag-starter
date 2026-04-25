# smart-rag-starter

A small, dependency-light Retrieval Augmented Generation starter kit in
Python: load, chunk, retrieve, prompt.

The goal is to give you a clean, readable baseline you can extend. The runtime
is **pure standard library** (no numpy, no sklearn, no extra deps), so it
installs in seconds and runs anywhere Python 3.10+ runs.

## Pipeline

```
+--------+     +---------+     +-----------+     +----------+
|  load  | --> |  chunk  | --> |  retrieve | --> |  prompt  |
| (docs) |     | (text)  |     |  (BM25)   |     | (citations)
+--------+     +---------+     +-----------+     +----------+
```

## Install

```bash
pip install -e .
# with dev deps for running the test suite:
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick start

```python
from smart_rag import BM25Retriever, build_prompt, chunk, load_documents

docs    = load_documents(["docs/*.md"])
chunks  = [c for d in docs for c in chunk(d, max_chars=800, overlap=100)]
hits    = BM25Retriever().fit(chunks).search("refund policy", k=5)
prompt  = build_prompt("refund policy", [c for c, _ in hits])
print(prompt)
```

## CLI

```bash
srag retrieve "your question" docs/*.md
srag retrieve "your question" docs/ --k 5 --max-chars 800 --overlap 100
srag retrieve "your question" docs/ --json   # machine-readable output
srag --version
srag --help
```

The `retrieve` subcommand:

- loads files (`.md`, `.txt`, `.py`, `.rst`) from any mix of paths, globs, and
  directories,
- chunks them into overlapping char windows,
- ranks chunks with BM25,
- prints a prompt with numbered citations and a Sources section (or a JSON
  payload with `--json`).

## Public API

| Symbol | What it does |
| --- | --- |
| `load_documents(paths)` | Read text files (md/txt/py/rst) into `Document` objects. Supports globs and directories. |
| `chunk(doc, max_chars=800, overlap=100)` | Slide a fixed-size window over `doc.text` with overlap; returns a list of `Chunk`. |
| `BM25Retriever(k1=1.5, b=0.75)` | Pure-Python BM25. `.fit(chunks)` then `.search(query, k=5)` returns `[(Chunk, score), ...]`. |
| `build_prompt(query, retrieved, *, system=None)` | Single-string prompt with `[1]`-style inline citations and a Sources section. |
| `build_messages(query, retrieved, *, system=None)` | Same content as `build_prompt`, shaped for chat APIs (`[{role, content}]`). |

## Why so minimal?

This starter kit is intentionally dep-light. It is meant to be:

- a teaching artifact you can read end-to-end in 15 minutes,
- a baseline you can A/B against fancier retrievers,
- a drop-in skeleton you fork before reaching for vectors / LLM clients.

## Roadmap

- Sentence/paragraph-aware chunker (split on boundaries, then rejoin to fit
  `max_chars`).
- Pluggable scorers (TF-IDF, hybrid BM25 + dense via optional extra).
- Persistence (sqlite) for the index, with incremental updates.
- Streaming evaluators and grounded-answer checks.
- Optional integrations (sentence-transformers, OpenAI/Anthropic clients) as
  extras, never as runtime requirements.

## License

MIT. See [LICENSE](LICENSE).
