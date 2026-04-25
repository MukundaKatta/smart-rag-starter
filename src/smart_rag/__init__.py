"""smart-rag-starter: a small, dependency-light Retrieval Augmented Generation starter kit.

Public API:
    - load_documents: read text files into Document objects
    - chunk: split a Document into overlapping char-window Chunks
    - BM25Retriever: pure-Python BM25 retriever over Chunks
    - build_prompt: assemble a numbered-citation prompt from retrieved chunks
"""

from .chunker import Chunk, chunk
from .loader import Document, load_documents
from .prompt import build_messages, build_prompt
from .retriever import BM25Retriever

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Document",
    "build_messages",
    "build_prompt",
    "chunk",
    "load_documents",
]

__version__ = "0.1.0"
