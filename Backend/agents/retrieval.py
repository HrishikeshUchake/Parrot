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

_MESSAGE_TOPIC_KEYWORDS = [
    "work",
    "music",
    "book",
    "fitness",
    "health",
    "weekend",
    "food",
    "tech",
    "citylife",
    "mindset",
    "learning",
]


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


def _format_ranked(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return "none found"
    return ", ".join(f"{name} ({count})" for name, count in rows)


def _top_message_partners(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (m:Message)
        WHERE m.user_context_username = $username
          AND (m.sender_name = $username OR m.receiver_name = $username)
        WITH CASE
            WHEN m.sender_name = $username THEN m.receiver_name
            ELSE m.sender_name
        END AS partner
        WHERE partner IS NOT NULL
          AND trim(partner) <> ""
          AND partner <> $username
        RETURN partner AS name, count(*) AS c
        ORDER BY c DESC, name ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(cypher, username=username,
                           limit=limit).data()  # noqa: SLF001
    return [(str(r.get("name", "")), int(r.get("c", 0))) for r in rows]


def _top_message_topics(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (m:Message)
        WHERE m.user_context_username = $username
          AND (m.sender_name = $username OR m.receiver_name = $username)
        WITH toLower(coalesce(m.text, "")) AS txt
        UNWIND $keywords AS kw
        WITH kw, txt
        WHERE txt CONTAINS kw
        RETURN kw AS topic, count(*) AS c
        ORDER BY c DESC, topic ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(
            cypher,
            username=username,
            keywords=_MESSAGE_TOPIC_KEYWORDS,
            limit=limit,
        ).data()  # noqa: SLF001
    return [(str(r.get("topic", "")), int(r.get("c", 0))) for r in rows]


def _top_engagers(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_COMMENT]->(c:Comment)
        WHERE c.commenter_name IS NOT NULL
          AND trim(c.commenter_name) <> ""
          AND c.commenter_name <> $username
        RETURN c.commenter_name AS name, count(*) AS c
        ORDER BY c DESC, name ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(cypher, username=username,
                           limit=limit).data()  # noqa: SLF001
    return [(str(r.get("name", "")), int(r.get("c", 0))) for r in rows]


def _top_themes_authored(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_TAG]->(t:Tag)
        WHERE p.account_username = $username
        RETURN toLower(t.name) AS tag, count(*) AS c
        ORDER BY c DESC, tag ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(cypher, username=username,
                           limit=limit).data()  # noqa: SLF001
    return [(str(r.get("tag", "")), int(r.get("c", 0))) for r in rows]


def _top_themes_interactions(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_TAG]->(t:Tag)
        WHERE p.account_username = $username
           OR EXISTS {
               MATCH (p)-[:HAS_COMMENT]->(c:Comment {commenter_name: $username})
           }
        RETURN toLower(t.name) AS tag, count(*) AS c
        ORDER BY c DESC, tag ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(cypher, username=username,
                           limit=limit).data()  # noqa: SLF001

    ranked = [(str(r.get("tag", "")), int(r.get("c", 0))) for r in rows]
    if ranked:
        return ranked

    fallback = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_TAG]->(t:Tag)
        RETURN toLower(t.name) AS tag, count(*) AS c
        ORDER BY c DESC, tag ASC
        LIMIT $limit
    """
    with _store._driver.session(database=_store._db) as session:
        rows = session.run(fallback, username=username,
                           limit=limit).data()  # noqa: SLF001
    return [(str(r.get("tag", "")), int(r.get("c", 0))) for r in rows]


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

    metadata_filter = state.get("filters") or None
    if user_context:
        metadata_filter = dict(metadata_filter or {})
        metadata_filter["user_context_username"] = user_context

    results = _engine.search(
        query=state["query"],
        top_k=settings.default_top_k,
        metadata_filter=metadata_filter,
    )

    # Add user-scoped message/comment semantic hits if user context is available.
    if user_context:
        query_embedding = _embedder.encode(state["query"])
        message_hits = _store.similarity_search_messages(
            query_embedding=query_embedding,
            username=user_context,
            top_k=max(3, settings.default_top_k // 2),
        )
        comment_hits = _store.similarity_search_comments(
            query_embedding=query_embedding,
            username=user_context,
            top_k=max(3, settings.default_top_k // 2),
        )
        results.extend(_to_message_result(h) for h in message_hits)
        results.extend(_to_comment_result(h) for h in comment_hits)

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

    # Step 1 – hybrid multi-query retrieval
    results: list[SearchResult] = []
    per_query_k = max(settings.advanced_top_k // max(1, len(queries)), 3)

    for q in queries:
        metadata_filter = {
            "user_context_username": user_context} if user_context else None
        results.extend(
            _engine.search(
                query=q,
                top_k=per_query_k,
                metadata_filter=metadata_filter,
            )
        )

    if user_context:
        # Add semantic matches from user comments/messages for each sub-query.
        for q in queries:
            emb = _embedder.encode(q)
            for h in _store.similarity_search_messages(
                query_embedding=emb,
                username=user_context,
                top_k=3,
            ):
                results.append(_to_message_result(h))
            for h in _store.similarity_search_comments(
                query_embedding=emb,
                username=user_context,
                top_k=3,
            ):
                results.append(_to_comment_result(h))

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


async def analytics_retrieval_node(state: AgentState) -> dict:
    """Deterministic aggregate analytics over user-scoped social graph data."""
    query = (state.get("query") or "").lower()
    user_context = _extract_user_context(state)

    if not user_context:
        return {
            "search_results": [],
            "answer": (
                "I can run analytics once you provide a user context, "
                "for example: 'for albert336'."
            ),
        }

    if "message the most" in query:
        top = _top_message_partners(user_context)
        if not top:
            answer = f"No direct message history found for user '{user_context}'."
        else:
            best_name, best_count = top[0]
            answer = (
                f"You message {best_name} the most ({best_count} messages). "
                f"Top message partners: {_format_ranked(top)}."
            )
        return {"search_results": [], "answer": answer}

    if "messages usually discuss" in query or "direct messages usually discuss" in query:
        top = _top_message_topics(user_context)
        if not top:
            answer = f"No recurring direct-message topics found for user '{user_context}'."
        else:
            answer = f"Your direct messages most often discuss: {_format_ranked(top)}."
        return {"search_results": [], "answer": answer}

    if "engage with my posts" in query or "engages with my posts" in query:
        top = _top_engagers(user_context)
        if not top:
            answer = f"No engagement comments found for user '{user_context}'."
        else:
            best_name, best_count = top[0]
            answer = (
                f"{best_name} engages with your posts the most ({best_count} comments). "
                f"Top engagers: {_format_ranked(top)}."
            )
        return {"search_results": [], "answer": answer}

    if "main topics i post" in query:
        top = _top_themes_authored(user_context)
        if not top:
            answer = f"No authored post themes found for user '{user_context}'."
        else:
            answer = f"Your main posting themes are: {_format_ranked(top)}."
        return {"search_results": [], "answer": answer}

    if "themes appear most" in query or "topics appear most" in query or "top themes" in query or "top topics" in query:
        top = _top_themes_interactions(user_context)
        if not top:
            answer = f"No recurring themes found in interacted posts for user '{user_context}'."
        else:
            answer = f"The most common themes in posts you interact with are: {_format_ranked(top)}."
        return {"search_results": [], "answer": answer}

    top = _top_themes_interactions(user_context)
    if top:
        answer = f"Top themes in your social graph are: {_format_ranked(top)}."
    else:
        answer = f"I couldn't find enough analytics data for user '{user_context}'."

    return {"search_results": [], "answer": answer}
