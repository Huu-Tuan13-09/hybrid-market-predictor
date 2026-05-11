"""
backend/rag_engine/retriever.py
================================
Context Retriever — semantic search over ChromaDB collections.

Responsibilities:
  1. Accept a natural-language query string.
  2. Embed the query and perform similarity search in ChromaDB.
  3. Post-process: filter low-similarity results, deduplicate by source.
  4. Format retrieved chunks into a clean context string for LLM prompts.

Design:
  - OOP, Type Hinting, try/except on every ChromaDB call.
  - Configurable n_results and similarity threshold per query.
  - Loguru for structured logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from loguru import logger

from backend.rag_engine.embedder import TextEmbedder
from backend.rag_engine.indexer import DEFAULT_CHROMA_PATH, CollectionName, ALL_COLLECTIONS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_N_RESULTS          = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.65   # Cosine similarity (0.0–1.0); lower = more permissive


# ---------------------------------------------------------------------------
# Result Data Class
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """Single retrieved chunk with its metadata and similarity score."""

    def __init__(
        self,
        text:       str,
        source:     str,
        date:       str,
        similarity: float,
        collection: str,
    ) -> None:
        self.text       = text
        self.source     = source
        self.date       = date
        self.similarity = similarity
        self.collection = collection

    def to_context_line(self) -> str:
        """Format as a single context line for LLM prompt injection."""
        return f"[{self.source} | {self.date} | sim={self.similarity:.2f}]\n{self.text}"


# ---------------------------------------------------------------------------
# ContextRetriever
# ---------------------------------------------------------------------------

class ContextRetriever:
    """
    Performs semantic search across one or multiple ChromaDB collections
    and formats results into LLM-ready context strings.

    Usage:
        retriever = ContextRetriever()

        # Query a specific collection
        context = retriever.query(
            query="Tác động của lãi suất FED đến thị trường Việt Nam",
            collection="macro_reports",
            n_results=5,
        )
        # context → formatted string for injection into agent prompt

        # Multi-collection query (returns merged results)
        context = retriever.multi_query(
            queries={
                "macro_reports":  "Vietnam monetary policy 2026",
                "market_news":    "VN-Index foreign investor May 2026",
            }
        )
    """

    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_PATH,
        embedder:    Optional[TextEmbedder] = None,
    ) -> None:
        chroma_path  = Path(chroma_path)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._embedder = embedder or TextEmbedder()
        logger.debug(f"ContextRetriever init | chroma_path={chroma_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query:      str,
        collection: CollectionName,
        n_results:  int   = DEFAULT_N_RESULTS,
        threshold:  float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> str:
        """
        Query a single collection and return a formatted context string.

        Args:
            query:      Natural-language query string.
            collection: ChromaDB collection name.
            n_results:  Maximum number of chunks to retrieve.
            threshold:  Minimum similarity score (0.0–1.0) to include a chunk.

        Returns:
            Formatted multi-line context string, or empty string if no results.
        """
        chunks = self._search(query, collection, n_results, threshold)
        if not chunks:
            logger.warning(f"No results above threshold={threshold} for query: '{query[:80]}'")
            return ""

        context = self._format_context(chunks, query)
        logger.debug(f"Retrieved {len(chunks)} chunks from '{collection}'")
        return context

    def query_raw(
        self,
        query:      str,
        collection: CollectionName,
        n_results:  int   = DEFAULT_N_RESULTS,
        threshold:  float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Query and return raw chunk dicts (for API responses, not LLM prompts).

        Returns:
            List of dicts with keys: text, source, date, similarity, collection.
        """
        chunks = self._search(query, collection, n_results, threshold)
        return [
            {
                "text":       c.text,
                "source":     c.source,
                "date":       c.date,
                "similarity": c.similarity,
                "collection": c.collection,
            }
            for c in chunks
        ]

    def multi_query(
        self,
        queries:   dict[CollectionName, str],
        n_results: int   = DEFAULT_N_RESULTS,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> str:
        """
        Query multiple collections and merge results into one context string.

        Args:
            queries: Dict mapping collection name → query string.
            n_results: Per-collection result limit.
            threshold: Minimum similarity threshold.

        Returns:
            Merged formatted context string with section headers.
        """
        sections: list[str] = []
        for collection, query in queries.items():
            chunks = self._search(query, collection, n_results, threshold)
            if chunks:
                header  = f"\n--- Nguồn: {collection.upper()} ---"
                content = "\n\n".join(c.to_context_line() for c in chunks)
                sections.append(f"{header}\n{content}")

        if not sections:
            return "Không tìm thấy ngữ cảnh vĩ mô phù hợp trong ChromaDB."

        return "\n".join(sections)

    def collection_count(self, collection: CollectionName) -> int:
        """Return number of documents in a collection. Returns 0 if not found."""
        try:
            col = self._client.get_collection(
                name=collection,
                embedding_function=self._embedder,
            )
            return col.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_collection(self, name: CollectionName) -> Optional[chromadb.Collection]:
        """Get an existing collection; return None if it doesn't exist yet."""
        try:
            return self._client.get_collection(
                name=name,
                embedding_function=self._embedder,
            )
        except Exception:
            logger.warning(f"Collection '{name}' does not exist in ChromaDB yet.")
            return None

    def _search(
        self,
        query:      str,
        collection: CollectionName,
        n_results:  int,
        threshold:  float,
    ) -> list[RetrievedChunk]:
        """Execute semantic search and apply threshold filter."""
        col = self._get_collection(collection)
        if col is None or col.count() == 0:
            return []

        try:
            results = col.query(
                query_texts     = [query],
                n_results       = min(n_results, col.count()),
                include         = ["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"ChromaDB query failed on '{collection}': {exc}")
            return []

        # ChromaDB returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity: sim = 1 - distance/2  (for cosine space)
        docs       = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            similarity = float(1.0 - dist / 2.0)
            if similarity < threshold:
                continue
            chunks.append(RetrievedChunk(
                text       = doc,
                source     = meta.get("source", "unknown"),
                date       = meta.get("date", ""),
                similarity = round(similarity, 4),
                collection = collection,
            ))

        # Deduplicate by source (keep highest similarity per source)
        seen_sources: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.source not in seen_sources or chunk.similarity > seen_sources[chunk.source].similarity:
                seen_sources[chunk.source] = chunk

        return list(seen_sources.values())

    def _format_context(self, chunks: list[RetrievedChunk], query: str) -> str:
        """
        Format retrieved chunks into a clean context string for LLM injection.

        Format:
          CONTEXT (truy xuất từ ChromaDB cho query: "..."):
          [source | date | sim=0.82]
          <text>

          [source | date | sim=0.77]
          <text>
        """
        header  = f'CONTEXT (truy xuất từ ChromaDB cho query: "{query[:100]}"):\n'
        body    = "\n\n".join(c.to_context_line() for c in chunks)
        return header + body
