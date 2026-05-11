"""
backend/rag_engine/embedder.py
================================
Text Embedder — wraps sentence-transformers for ChromaDB-compatible embeddings.

Uses 'all-MiniLM-L6-v2' (384-dim) as the default embedding model:
  - Small footprint (~80MB), runs on CPU without GPU requirement.
  - Excellent multilingual support — handles Vietnamese financial text well.
  - Compatible with ChromaDB's custom embedding function interface.

Design:
  - Singleton-friendly: model is loaded once and reused.
  - Implements ChromaDB's EmbeddingFunction protocol for seamless integration.
  - Supports both single string and batch list inputs.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings
from loguru import logger
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE  = 32
DEFAULT_NORMALIZE   = True


# ---------------------------------------------------------------------------
# Embedder Class
# ---------------------------------------------------------------------------

class TextEmbedder(EmbeddingFunction):
    """
    Wraps SentenceTransformer and implements ChromaDB's EmbeddingFunction
    protocol so it can be passed directly to ChromaDB collection constructors.

    Usage (standalone):
        embedder = TextEmbedder()
        vecs = embedder.embed_texts(["Lạm phát tăng cao gây áp lực lên VN-Index"])
        # vecs: list[list[float]], shape (1, 384)

    Usage (ChromaDB):
        client     = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_or_create_collection(
            name="macro_reports",
            embedding_function=TextEmbedder()
        )
    """

    def __init__(
        self,
        model_name:  str  = DEFAULT_EMBED_MODEL,
        batch_size:  int  = DEFAULT_BATCH_SIZE,
        normalize:   bool = DEFAULT_NORMALIZE,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier.
            batch_size: Batch size for encoding (larger = faster, more memory).
            normalize:  L2-normalize embeddings (recommended for cosine similarity).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize
        self._model: SentenceTransformer | None = None
        logger.debug(f"TextEmbedder init | model={model_name}")

    # ------------------------------------------------------------------
    # ChromaDB EmbeddingFunction protocol
    # ------------------------------------------------------------------

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB calls this method when indexing or querying.
        'input' is a list of strings (Documents = list[str]).
        Returns list of embedding vectors (list[list[float]]).
        """
        return self.embed_texts(list(input))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of input strings. Empty strings are replaced with a
                   whitespace to avoid model errors.

        Returns:
            List of float vectors, shape (len(texts), embedding_dim).
        """
        model = self._get_model()
        # Guard against empty strings
        safe_texts = [t if t.strip() else " " for t in texts]

        try:
            embeddings = model.encode(
                safe_texts,
                batch_size        = self.batch_size,
                normalize_embeddings = self.normalize,
                show_progress_bar = False,
                convert_to_numpy  = True,
            )
        except Exception as exc:
            logger.error(f"Embedding failed: {exc}")
            raise RuntimeError(f"TextEmbedder.embed_texts failed: {exc}") from exc

        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Convenience method for embedding a single string."""
        return self.embed_texts([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._get_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the model on first call."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.success(
                    f"Embedding model loaded | dim={self._model.get_sentence_embedding_dimension()}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model
