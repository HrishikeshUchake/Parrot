"""EmbeddingService wraps SentenceTransformer with LRU cache."""
from __future__ import annotations
import functools
from sentence_transformers import SentenceTransformer

from ..config import settings


class EmbeddingService:
    """Singleton-style embedding service with an in-process LRU cache."""

    _instance: "EmbeddingService | None" = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(
                settings.embedding_model)
            cls._instance._encode_cached = functools.lru_cache(
                maxsize=settings.embedding_cache_size
            )(cls._instance._encode_single)
        return cls._instance

    def encode(self, text: str) -> list[float]:
        """Return a normalised embedding for a single text (cached)."""
        return self._encode_cached(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Return normalised embeddings for a batch (no LRU for ingestion)."""
        vecs = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def _encode_single(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()
