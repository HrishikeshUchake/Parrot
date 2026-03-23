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
    """Build Post objects directly from Neo4j vector hit metadata."""

    def __init__(self, store: VectorStore, svc: EmbeddingService) -> None:
        self._store = store
        self._svc = svc

    def search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        where: dict | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> list[SearchResult]:
        emb = self._svc.encode(query)
        hits = self._store.similarity_search(
            emb, top_k=top_k, where=where or None, date_range=date_range
        )
        results = []

        # When filtering by metadata (e.g., specific author), use a lower threshold
        # since this is a metadata query, not a pure content similarity query
        threshold = 0.15 if where else settings.similarity_threshold

        for h in hits:
            if h["score"] < threshold:
                continue
            meta = h["metadata"]
            post = Post(
                id=h["id"],
                content=h["document"],
                created_at=meta.get("created_at", ""),
                account_id=meta.get("account_id", ""),
                account_username=meta.get("account_username", ""),
                account_display_name=meta.get("account_display_name", ""),
                account_acct=meta.get("account_acct", ""),
                tags=meta.get("tags", []),
                reblogs_count=int(meta.get("reblogs_count") or 0),
                favourites_count=int(meta.get("favourites_count") or 0),
                replies_count=int(meta.get("replies_count") or 0),
                url=meta.get("url", ""),
                visibility=meta.get("visibility", "public"),
                language=meta.get("language", ""),
            )
            results.append(
                SearchResult(
                    post=post,
                    result_type="post",
                    item_id=post.id,
                    content=post.content,
                    metadata={
                        "account_username": post.account_username,
                        "account_acct": post.account_acct,
                        "tags": post.tags,
                    },
                    score=h["score"],
                    source="vector",
                )
            )
        return results


class KeywordSearchStrategy:
    def __init__(self, repo: PostRepository) -> None:
        self._repo = repo

    def search(self, query: str, top_k: int = settings.default_top_k) -> list[SearchResult]:
        posts = self._repo.keyword_search(query, limit=top_k)
        results: list[SearchResult] = []
        for rank, p in enumerate(posts, start=1):
            results.append(
                SearchResult(
                    post=p,
                    result_type="post",
                    item_id=p.id,
                    content=p.content,
                    metadata={
                        "account_username": p.account_username,
                        "account_acct": p.account_acct,
                        "tags": p.tags,
                    },
                    # Per-source score; final fusion/reranking is done via RRF in HybridSearchEngine
                    score=1.0 / rank,
                    source="keyword",
                )
            )
        return results


class HybridSearchEngine:
    """
    Combines VectorSearchStrategy and KeywordSearchStrategy,
    deduplicates, and reranks by score.
    """

    def __init__(self) -> None:
        repo = PostRepository()
        svc = EmbeddingService()
        store = VectorStore()
        self._vector = VectorSearchStrategy(store, svc)
        self._keyword = KeywordSearchStrategy(repo)

    def search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
        metadata_filter: dict | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> list[SearchResult]:
        vec_results = self._vector.search(
            query, top_k=top_k, where=metadata_filter, date_range=date_range
        )
        kw_results = self._keyword.search(query, top_k=top_k)

        # Reciprocal Rank Fusion
        k_rrf = 60
        scores: dict[str, float] = {}
        merged_dict: dict[str, SearchResult] = {}
        vec_ids: set[str] = set()

        for rank, r in enumerate(vec_results, start=1):
            pid = r.post.id
            vec_ids.add(pid)
            if pid not in merged_dict:
                merged_dict[pid] = r
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_rrf + rank)

        for rank, r in enumerate(kw_results, start=1):
            pid = r.post.id
            if pid not in merged_dict:
                r.source = "keyword"
                merged_dict[pid] = r
            else:
                merged_dict[pid].source = "hybrid"
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_rrf + rank)

        merged: list[SearchResult] = []
        for pid, score in scores.items():
            result = merged_dict[pid]
            result.score = score
            merged.append(result)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def advanced_search(
        self,
        queries: list[str],
        top_k: int = settings.advanced_top_k,
        date_range: tuple[str, str] | None = None,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Multi-query retrieval: run multiple query reformulations, merge."""
        seen: set[str] = set()
        all_results: list[SearchResult] = []
        per_query_k = max(top_k // len(queries), 3)

        for q in queries:
            for r in self.search(q, top_k=per_query_k, date_range=date_range, metadata_filter=metadata_filter):
                if r.post.id not in seen:
                    seen.add(r.post.id)
                    all_results.append(r)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
