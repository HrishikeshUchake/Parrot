"""Retrieval nodes – simple lookup and advanced multi-query retrieval."""
from __future__ import annotations
import logging

from .state import AgentState
from ..config import settings
from ..database.models import SearchResult
from ..database.repository import PostRepository
from ..services.search_engine import HybridSearchEngine
from ..services.vector_store import VectorStore

logger = logging.getLogger(__name__)
_engine = HybridSearchEngine()
_store = VectorStore()
_repo = PostRepository()

_META_PATTERNS = ["how many posts", "total posts", "count of posts", "database size"]


async def simple_retrieval_node(state: AgentState) -> dict:
    """Vector + keyword hybrid search for a single query."""
    query_lower = state["query"].lower()

    # Handle meta-queries directly without retrieval
    if any(p in query_lower for p in _META_PATTERNS):
        count = _store.count
        return {
            "search_results": [],
            "answer": f"There are {count} posts in the database.",
        }

    results = _engine.search(
        query=state["query"],
        top_k=settings.default_top_k,
        metadata_filter=state.get("filters") or None,
    )
    logger.info("Simple retrieval: %d results", len(results))
    return {"search_results": results}


async def advanced_retrieval_node(state: AgentState) -> dict:
    """
    Multi-query retrieval with graph-neighbor enrichment.

    Steps:
      1. Multi-query hybrid search (vector + keyword across all sub-queries).
      2. For each top-scored post, traverse Neo4j graph edges (shared tags /
         authorship) to surface strongly related posts the vector search may
         have missed.
      3. Merge, deduplicate, and rerank.
    """
    queries: list[str] = list(state.get("sub_queries") or [])
    if state["query"] not in queries:
        queries = [state["query"]] + queries

    # Step 1 – hybrid multi-query retrieval
    results = _engine.advanced_search(queries=queries, top_k=settings.advanced_top_k)
    logger.info("Advanced retrieval step 1: %d results from %d queries", len(results), len(queries))

    # Step 2 – graph-neighbor enrichment via Neo4j traversal
    seen_ids = {r.post.id for r in results}
    neighbor_post_ids: list[int] = []

    for r in results[: settings.default_top_k]:          # expand from top-k seeds
        neighbors = _store.graph_neighbors(r.post.id, hops=1)
        for n in neighbors:
            nid = int(n["id"])
            if nid not in seen_ids:
                seen_ids.add(nid)
                neighbor_post_ids.append(nid)

    if neighbor_post_ids:
        neighbor_posts = _repo.get_by_ids(neighbor_post_ids)
        graph_results = [
            SearchResult(post=p, score=0.5, source="graph")   # graph signal score
            for p in neighbor_posts
        ]
        results = results + graph_results
        logger.info("Graph enrichment added %d neighbor posts.", len(graph_results))

    # Step 3 – rerank by score, keep top-k
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[: settings.advanced_top_k]

    logger.info("Advanced retrieval final: %d results", len(results))
    return {"search_results": results}
