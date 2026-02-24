"""HybridSearchEngine – Strategy pattern combining vector + keyword search."""
from __future__ import annotations
from typing import Protocol

from ..config import settings
from ..database.models import Post, SearchResult
from ..database.repository import PostRepository
from .embedding_service import EmbeddingService
from .vector_store import VectorStore


class SearchStrategy(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        ...


# ── Concrete strategies ───────────────────────────────────────────────────────

class VectorSearchStrategy:
    def __init__(self, store: VectorStore, repo: PostRepository, svc: EmbeddingService) -> None:
        self._store = store
        self._repo = repo
        self._svc = svc

    def search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        where: dict | None = None,
    ) -> list[SearchResult]:
        emb = self._svc.encode(query)
        hits = self._store.similarity_search(emb, top_k=top_k, where=where or None)
        if not hits:
            return []
        post_ids = [h["id"] for h in hits]
        posts_by_id = {p.id: p for p in self._repo.get_by_ids(post_ids)}
        results = []
        for h in hits:
            if h["score"] < settings.similarity_threshold:
                continue
            post = posts_by_id.get(h["id"])
            if post:
                results.append(SearchResult(post=post, score=h["score"], source="vector"))
        return results


class KeywordSearchStrategy:
    def __init__(self, repo: PostRepository) -> None:
        self._repo = repo

    def search(self, query: str, top_k: int = settings.default_top_k) -> list[SearchResult]:
        posts = self._repo.keyword_search(query, limit=top_k)
        return [SearchResult(post=p, score=1.0, source="keyword") for p in posts]


class HybridSearchEngine:
    """
    Combines VectorSearchStrategy and KeywordSearchStrategy,
    deduplicates, and reranks by score.
    """

    def __init__(self) -> None:
        repo = PostRepository()
        svc = EmbeddingService()
        store = VectorStore()
        self._vector = VectorSearchStrategy(store, repo, svc)
        self._keyword = KeywordSearchStrategy(repo)

    def search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        vec_results = self._vector.search(query, top_k=top_k, where=metadata_filter)
        kw_results = self._keyword.search(query, top_k=top_k)

        seen: set[int] = set()
        merged: list[SearchResult] = []

        for r in vec_results:
            if r.post.id not in seen:
                seen.add(r.post.id)
                merged.append(r)

        for r in kw_results:
            if r.post.id not in seen:
                seen.add(r.post.id)
                r.source = "keyword"
                merged.append(r)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def advanced_search(
        self, queries: list[str], top_k: int = settings.advanced_top_k
    ) -> list[SearchResult]:
        """Multi-query retrieval: run multiple query reformulations, merge."""
        seen: set[int] = set()
        all_results: list[SearchResult] = []
        per_query_k = max(top_k // len(queries), 3)

        for q in queries:
            for r in self.search(q, top_k=per_query_k):
                if r.post.id not in seen:
                    seen.add(r.post.id)
                    all_results.append(r)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
