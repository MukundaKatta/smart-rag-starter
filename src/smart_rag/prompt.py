"""Prompt assembly for RAG.

Given a query and a list of retrieved chunks, build a single string prompt
(or a list of chat messages) that includes inline numbered citations and a
trailing "Sources" section listing each chunk's origin.
"""

from __future__ import annotations

from typing import Iterable

from .chunker import Chunk


DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context snippets. Cite sources inline using their bracketed "
    "numbers, for example [1]. If the answer is not in the context, say so."
)


def _format_context(retrieved: Iterable[Chunk]) -> tuple[str, str]:
    """Return (context_block, sources_block) for the given chunks."""
    context_lines: list[str] = []
    source_lines: list[str] = []
    for i, ch in enumerate(retrieved, start=1):
        context_lines.append(f"[{i}] {ch.text}")
        source_lines.append(f"[{i}] {ch.source}")
    return "\n\n".join(context_lines), "\n".join(source_lines)


def build_prompt(
    query: str,
    retrieved: list[Chunk],
    *,
    system: str | None = None,
) -> str:
    """Assemble a single-string prompt with numbered citations and Sources.

    Args:
        query: The user's question.
        retrieved: Chunks returned by a retriever, in rank order.
        system: Optional override for the system instruction.
    """
    sys_block = system if system is not None else DEFAULT_SYSTEM
    context_block, sources_block = _format_context(retrieved)

    parts = [
        f"System: {sys_block}",
        "Context:",
        context_block if context_block else "(no context retrieved)",
        f"Question: {query}",
        "Answer (cite sources inline like [1]):",
        "",
        "Sources:",
        sources_block if sources_block else "(none)",
    ]
    return "\n\n".join(parts)


def build_messages(
    query: str,
    retrieved: list[Chunk],
    *,
    system: str | None = None,
) -> list[dict]:
    """Return a chat-style [{role, content}] list for OpenAI-shaped APIs."""
    sys_block = system if system is not None else DEFAULT_SYSTEM
    context_block, sources_block = _format_context(retrieved)

    user_content = (
        "Context:\n\n"
        + (context_block if context_block else "(no context retrieved)")
        + f"\n\nQuestion: {query}\n\n"
        + "Answer (cite sources inline like [1]).\n\n"
        + "Sources:\n"
        + (sources_block if sources_block else "(none)")
    )
    return [
        {"role": "system", "content": sys_block},
        {"role": "user", "content": user_content},
    ]
