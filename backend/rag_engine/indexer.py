"""
backend/rag_engine/indexer.py
================================
Document Indexer — ingests raw documents into ChromaDB collections.

Pipeline:
  1. Accept file paths (PDF, TXT, DOCX) or raw text strings.
  2. Parse documents via 'unstructured' library.
  3. Chunk text with RecursiveCharacterTextSplitter logic.
  4. Embed chunks via TextEmbedder.
  5. Upsert into ChromaDB with metadata (source, date, collection).

Collections managed:
  - macro_reports : GDP, CPI, FED decisions, NHNN policy documents
  - market_news   : Daily crawled news articles (from NewsScraper)
  - company_filings: Quarterly/annual financial reports

Can be run as a standalone script to seed the vector DB on first setup:
    python backend/rag_engine/indexer.py

Design:
  - OOP, Type Hinting, try/except per document so one failure doesn't abort batch.
  - Loguru for structured logging.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import chromadb
from loguru import logger

from backend.rag_engine.embedder import TextEmbedder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CHROMA_PATH  = Path("data/chromadb")
CHUNK_SIZE           = 512    # characters
CHUNK_OVERLAP        = 64     # characters
MIN_CHUNK_LENGTH     = 50     # discard chunks shorter than this

CollectionName = Literal["macro_reports", "market_news", "company_filings"]
ALL_COLLECTIONS: list[CollectionName] = ["macro_reports", "market_news", "company_filings"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a single text chunk ready for embedding and indexing."""
    text:       str
    doc_id:     str                      # SHA-256 of (source + text) for dedup
    source:     str                      # Filename or URL
    collection: CollectionName
    date:       str = field(default_factory=lambda: str(date.today()))
    chunk_idx:  int = 0

    @classmethod
    def create(
        cls,
        text:       str,
        source:     str,
        collection: CollectionName,
        chunk_idx:  int = 0,
        doc_date:   str = "",
    ) -> "DocumentChunk":
        doc_id = hashlib.sha256(f"{source}::{chunk_idx}::{text[:64]}".encode()).hexdigest()[:16]
        return cls(
            text       = text,
            doc_id     = doc_id,
            source     = source,
            collection = collection,
            date       = doc_date or str(date.today()),
            chunk_idx  = chunk_idx,
        )


# ---------------------------------------------------------------------------
# Text Splitter (pure Python — no LangChain dependency)
# ---------------------------------------------------------------------------

def _recursive_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks using sentence boundaries.

    Strategy:
      1. Split on paragraph boundaries first (\n\n).
      2. If a paragraph is still too large, split on sentence boundaries.
      3. Apply sliding window with overlap.
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) <= chunk_size:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
                # Start next chunk with overlap from end of previous buffer
                buffer = buffer[-overlap:].strip() + " " + para
            else:
                # Single paragraph exceeds chunk_size — split by sentences
                sentences = re.split(r"(?<=[.!?…])\s+", para)
                for sent in sentences:
                    if len(buffer) + len(sent) <= chunk_size:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = buffer[-overlap:].strip() + " " + sent if buffer else sent

    if buffer:
        chunks.append(buffer)

    # Filter minimum length
    return [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------

class DocumentIndexer:
    """
    Ingests documents into ChromaDB collections with automatic chunking
    and deduplication via content-based IDs.

    Usage:
        indexer = DocumentIndexer()

        # Index a PDF file
        indexer.index_file(
            path=Path("data/macro_report_Q1_2026.pdf"),
            collection="macro_reports",
        )

        # Index news articles from NewsScraper
        indexer.index_news_articles(articles)

        # Index raw text
        indexer.index_text(
            text="GDP Việt Nam Q1 2026 tăng 6.5%...",
            source="manual_input",
            collection="macro_reports",
        )
    """

    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_PATH,
        embedder:    Optional[TextEmbedder] = None,
    ) -> None:
        chroma_path = Path(chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._client   = chromadb.PersistentClient(path=str(chroma_path))
        self._embedder = embedder or TextEmbedder()
        logger.debug(f"DocumentIndexer init | chroma_path={chroma_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_file(
        self,
        path:       Path,
        collection: CollectionName,
        doc_date:   str = "",
    ) -> int:
        """
        Parse and index a single file (PDF, TXT, DOCX, MD).

        Returns:
            Number of chunks successfully indexed.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return 0

        logger.info(f"Indexing file: {path.name} → collection='{collection}'")
        text = self._extract_text(path)
        if not text:
            logger.warning(f"No text extracted from {path.name}")
            return 0

        return self.index_text(
            text       = text,
            source     = path.name,
            collection = collection,
            doc_date   = doc_date,
        )

    def index_directory(
        self,
        dir_path:   Path,
        collection: CollectionName,
        glob:       str = "**/*.{pdf,txt,md,docx}",
    ) -> int:
        """Index all matching files in a directory recursively."""
        dir_path = Path(dir_path)
        total    = 0
        patterns = ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.docx"]
        for pattern in patterns:
            for file_path in dir_path.glob(pattern):
                total += self.index_file(file_path, collection)
        logger.success(f"Directory indexing complete: {total} chunks from '{dir_path}'")
        return total

    def index_text(
        self,
        text:       str,
        source:     str,
        collection: CollectionName,
        doc_date:   str = "",
    ) -> int:
        """
        Chunk and index a raw text string.

        Returns:
            Number of chunks indexed.
        """
        chunks_text = _recursive_split(text)
        if not chunks_text:
            logger.warning(f"No chunks produced from source='{source}'")
            return 0

        chunks = [
            DocumentChunk.create(
                text       = chunk,
                source     = source,
                collection = collection,
                chunk_idx  = i,
                doc_date   = doc_date,
            )
            for i, chunk in enumerate(chunks_text)
        ]
        return self._upsert_chunks(chunks)

    def index_news_articles(self, articles: list[dict]) -> int:
        """
        Convenience method: index a list of article dicts from NewsScraper.

        Each article dict must have keys: title, summary, url, source, published_at.
        Combines title + summary into a single indexable text.
        """
        total = 0
        for article in articles:
            try:
                text = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
                n    = self.index_text(
                    text       = text,
                    source     = article.get("url", article.get("source", "news")),
                    collection = "market_news",
                    doc_date   = article.get("published_at", ""),
                )
                total += n
            except Exception as exc:
                logger.warning(f"Failed to index article '{article.get('title', '')}': {exc}")
        logger.success(f"Indexed {total} chunks from {len(articles)} news articles")
        return total

    def get_collection_stats(self) -> dict[str, int]:
        """Return document count for each registered collection."""
        stats = {}
        for name in ALL_COLLECTIONS:
            try:
                col    = self._get_or_create_collection(name)
                stats[name] = col.count()
            except Exception:
                stats[name] = -1
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> str:
        """Extract plain text from a file using 'unstructured' or plain read."""
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(filename=str(path))
                return "\n\n".join(str(el) for el in elements if str(el).strip())
            elif suffix == ".docx":
                from unstructured.partition.docx import partition_docx
                elements = partition_docx(filename=str(path))
                return "\n\n".join(str(el) for el in elements if str(el).strip())
            else:
                # TXT, MD, or unknown → plain read
                return path.read_text(encoding="utf-8", errors="ignore")
        except ImportError:
            logger.warning("'unstructured' not available for this file type — reading as plain text")
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.error(f"Could not read file {path}: {exc}")
                return ""
        except Exception as exc:
            logger.error(f"Text extraction failed for {path}: {exc}")
            return ""

    def _get_or_create_collection(self, name: CollectionName) -> chromadb.Collection:
        """Get or create a ChromaDB collection with the shared embedder."""
        return self._client.get_or_create_collection(
            name               = name,
            embedding_function = self._embedder,
            metadata           = {"hnsw:space": "cosine"},
        )

    def _upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Batch-upsert chunks into their respective ChromaDB collection."""
        if not chunks:
            return 0

        collection = self._get_or_create_collection(chunks[0].collection)

        ids        = [c.doc_id    for c in chunks]
        documents  = [c.text      for c in chunks]
        metadatas  = [{"source": c.source, "date": c.date, "chunk_idx": c.chunk_idx}
                      for c in chunks]

        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.debug(
                f"Upserted {len(chunks)} chunks → collection='{chunks[0].collection}'"
            )
            return len(chunks)
        except Exception as exc:
            logger.error(f"ChromaDB upsert failed: {exc}")
            return 0


# ---------------------------------------------------------------------------
# CLI Entry Point — Seed vector DB on first run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.add(sys.stdout, level="INFO")

    chroma_path = Path(os.getenv("CHROMA_DB_PATH", str(DEFAULT_CHROMA_PATH)))
    indexer     = DocumentIndexer(chroma_path=chroma_path)

    # Seed from any files in data/seed_documents/ if present
    seed_dir = Path("data/seed_documents")
    if seed_dir.exists():
        logger.info(f"Seeding from {seed_dir}…")
        indexer.index_directory(seed_dir, collection="macro_reports")
    else:
        logger.warning(
            f"No seed documents found at '{seed_dir}'. "
            "Place PDF/TXT macro reports there and re-run this script."
        )

    stats = indexer.get_collection_stats()
    logger.info(f"ChromaDB collection stats: {stats}")
