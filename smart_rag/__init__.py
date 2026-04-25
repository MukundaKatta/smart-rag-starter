"""smart-rag-starter: a small, dependency-light RAG starter kit."""

from .chunker import Chunk, Chunker
from .document import Document, load_directory, load_file
from .pipeline import RAGPipeline, RAGResult
from .retriever import Hit, Retriever, TfidfRetriever

__all__ = [
    "Chunk",
    "Chunker",
    "Document",
    "Hit",
    "RAGPipeline",
    "RAGResult",
    "Retriever",
    "TfidfRetriever",
    "load_directory",
    "load_file",
]

__version__ = "0.1.0"
