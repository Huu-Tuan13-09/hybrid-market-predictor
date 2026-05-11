"""
backend/rag_engine/__init__.py
RAG Engine package — ChromaDB vector store with semantic retrieval.

Exports:
  - TextEmbedder   : HuggingFace sentence-transformers wrapper
  - DocumentIndexer: Ingest and index documents into ChromaDB
  - ContextRetriever: Semantic query → formatted context string
"""

from .embedder import TextEmbedder
from .indexer import DocumentIndexer
from .retriever import ContextRetriever

__all__ = ["TextEmbedder", "DocumentIndexer", "ContextRetriever"]
