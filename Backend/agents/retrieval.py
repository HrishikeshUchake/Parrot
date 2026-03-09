"""Retrieval nodes simple lookup and advanced multi-query retrieval."""
from __future__ import annotations
import logging
import re

from .state import AgentState
from ..config import settings
from ..database.models import SearchResult
from ..database.repository import PostRepository
from ..services.embedding_service import EmbeddingService
from ..services.search_engine import HybridSearchEngine
from ..services.vector_store import VectorStore

logger = logging.getLogger(__name__)
_engine = HybridSearchEngine()
_store = VectorStore()
_repo = PostRepository()
_embedder = EmbeddingService()

_META_PATTERNS = ["how many posts", "total posts",
                  "count of posts", "database size"]


def _extract_user_context(state: AgentState) -> str:
    """Get username from explicit state/filter first, then fallback to query hints."""
    explicit = (state.get("user_context_username") or "").strip()
    if explicit:
        return explicit

    filters = state.get("filters") or {}
    for key in ("user_context_username", "username"):
        val = str(filters.get(key, "")).strip()
        if val:
            return val

    # Lightweight heuristic for queries like "for albert336", "user @albert336", "by riko"
    m = re.search(
        r"(?:for|user|about|by|from)\s+@?([a-zA-Z0-9_]{3,})", state.get("query", ""))
    return m.group(1) if m else ""


def _detect_content_types(query: str) -> dict[str, bool]:
    """
    Detect what content types the user is asking about.
    Returns dict with keys: posts, messages, comments
    """
    query_lower = query.lower()
    
    # Check for explicit mentions of each type (include common synonyms)
    has_posts = any(tok in query_lower for tok in ("post", "status", "toot"))
    has_messages = any(tok in query_lower for tok in (
        "message", "messages", "dm", "dms", "chat", "inbox"
    ))
    has_comments = any(tok in query_lower for tok in (
        "comment", "comments", "reply", "replies", "thread"
    ))
    
    # If none are explicitly mentioned, default to posts only
    # (to maintain backward compatibility and avoid mixing content types)
    if not (has_posts or has_messages or has_comments):
        has_posts = True
    
    return {
        "posts": has_posts,
        "messages": has_messages,
        "comments": has_comments,
    }


def _to_message_result(hit: dict) -> SearchResult:
    return SearchResult(
        post=None,
        result_type="message",
        item_id=str(hit.get("id", "")),
        content=hit.get("document", ""),
        metadata=hit.get("metadata", {}),
        score=float(hit.get("score", 0.0)),
        source="vector",
    )


def _to_comment_result(hit: dict) -> SearchResult:
    return SearchResult(
        post=None,
        result_type="comment",
        item_id=str(hit.get("id", "")),
        content=hit.get("document", ""),
        metadata=hit.get("metadata", {}),
        score=float(hit.get("score", 0.0)),
        source="vector",
    )


async def simple_retrieval_node(state: AgentState) -> dict:
    """Vector + keyword hybrid search for a single query."""
    query_lower = state["query"].lower()
    user_context = _extract_user_context(state)

    # Handle meta-queries directly without retrieval
    if any(p in query_lower for p in _META_PATTERNS):
        count = _store.count_for_user(
            user_context) if user_context else _store.count
        scope = f" for user '{user_context}'" if user_context else ""
        return {
            "search_results": [],
            "answer": f"There are {count} posts in the database{scope}.",
        }

    # Detect what content types the user is asking for
    content_types = _detect_content_types(state["query"])
    date_filter = state.get("date_filter")

    results: list[SearchResult] = []

    if content_types["posts"]:
        metadata_filter = state.get("filters") or None
        if user_context:
            # When asking for posts "by" a user, filter by author (account_username)
            metadata_filter = dict(metadata_filter or {})
            metadata_filter["account_username"] = user_context

        results.extend(
            _engine.search(
                query=state["query"],
                top_k=settings.default_top_k,
                metadata_filter=metadata_filter,
                date_range=date_filter,
            )
        )

    # Add user-scoped message/comment semantic hits only if explicitly requested
    if user_context:
        query_embedding = _embedder.encode(state["query"])
        
        # Use larger k when query only asks for messages/comments and excludes posts.
        semantic_k = settings.default_top_k if not content_types["posts"] else max(
            3, settings.default_top_k // 2
        )

        if content_types["messages"]:
            message_hits = _store.similarity_search_messages(
                query_embedding=query_embedding,
                username=user_context,
                top_k=semantic_k,
            )
            results.extend(_to_message_result(h) for h in message_hits)
        
        if content_types["comments"]:
            comment_hits = _store.similarity_search_comments(
                query_embedding=query_embedding,
                username=user_context,
                top_k=semantic_k,
            )
            results.extend(_to_comment_result(h) for h in comment_hits)

    # If the query explicitly asks only for messages/comments, avoid polluting
    # the final context with posts.
    if not content_types["posts"]:
        results = [
            r for r in results
            if (content_types["messages"] and r.result_type == "message")
            or (content_types["comments"] and r.result_type == "comment")
        ]

    results.sort(key=lambda x: x.score, reverse=True)
    results = results[: settings.default_top_k]
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
    user_context = _extract_user_context(state)

    # Detect what content types the user is asking for
    content_types = _detect_content_types(state["query"])

    # Step 1 – hybrid multi-query retrieval
    results: list[SearchResult] = []
    per_query_k = max(settings.advanced_top_k // max(1, len(queries)), 3)
    date_filter = state.get("date_filter")

    if content_types["posts"]:
        for q in queries:
            # When asking for posts "by" a user, filter by author (account_username)
            metadata_filter = None
            if user_context:
                metadata_filter = {"account_username": user_context}
            results.extend(
                _engine.search(
                    query=q,
                    top_k=per_query_k,
                    metadata_filter=metadata_filter,
                    date_range=date_filter,
                )
            )

    if user_context:
        # Add semantic matches from user comments/messages only if explicitly requested
        for q in queries:
            emb = _embedder.encode(q)
            
            semantic_k = 6 if content_types["posts"] else settings.default_top_k

            if content_types["messages"]:
                for h in _store.similarity_search_messages(
                    query_embedding=emb,
                    username=user_context,
                    top_k=semantic_k,
                ):
                    results.append(_to_message_result(h))
            
            if content_types["comments"]:
                for h in _store.similarity_search_comments(
                    query_embedding=emb,
                    username=user_context,
                    top_k=semantic_k,
                ):
                    results.append(_to_comment_result(h))

    if not content_types["posts"]:
        results = [
            r for r in results
            if (content_types["messages"] and r.result_type == "message")
            or (content_types["comments"] and r.result_type == "comment")
        ]

    # Deduplicate by (type, id), keeping best score.
    dedup: dict[tuple[str, str], SearchResult] = {}
    for r in results:
        key = (r.result_type, r.item_id)
        if key not in dedup or r.score > dedup[key].score:
            dedup[key] = r
    results = list(dedup.values())

    logger.info("Advanced retrieval step 1: %d results from %d queries", len(
        results), len(queries))

    # Step 2 – graph-neighbor enrichment via Neo4j traversal
    if content_types["posts"]:
        seen_ids = {r.post.id for r in results if r.post is not None}
        neighbor_post_ids: list[str] = []

        for r in results[: settings.default_top_k]:          # expand from top-k seeds
            if r.post is None:
                continue
            neighbors = _store.graph_neighbors(r.post.id, hops=1)
            for n in neighbors:
                nid = str(n["id"])
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    neighbor_post_ids.append(nid)

        if neighbor_post_ids:
            neighbor_posts = _repo.get_by_ids(neighbor_post_ids)
            graph_results = [
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
                    score=0.5,
                    source="graph",
                )
                for p in neighbor_posts
            ]
            results = results + graph_results
            logger.info("Graph enrichment added %d neighbor posts.",
                        len(graph_results))

    # Step 3 – rerank by score, keep top-k
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[: settings.advanced_top_k]

    logger.info("Advanced retrieval final: %d results", len(results))
    return {"search_results": results}
