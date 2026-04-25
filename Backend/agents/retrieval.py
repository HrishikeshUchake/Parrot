"""Retrieval nodes simple lookup and advanced multi-query retrieval."""
from __future__ import annotations
import logging
import re
from datetime import datetime, timedelta, timezone

from .state import AgentState
from ..config import settings
from ..database.models import SearchResult, ConversationThread
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


def _to_thread_result(hit: dict) -> SearchResult:
    meta = hit.get("metadata", {})
    thread_obj = None
    if meta.get("type") == "thread":
        thread_obj = ConversationThread(
            id=str(hit.get("id", "")),
            source_type=meta.get("source_type", "post"),
            participants=meta.get("participants", []),
            messages=meta.get("messages", []),
            summary=meta.get("summary", ""),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", "")
        )
    return SearchResult(
        post=None,
        thread=thread_obj,
        result_type="thread",
        item_id=str(hit.get("id", "")),
        content=hit.get("document", ""),
        metadata=meta,
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


def _safe_iso_date(value: str) -> str:
    """Normalize an ISO timestamp into YYYY-MM-DD for Cypher date() use.

    Assumes the input is an ISO 8601 date or datetime string. If the value is
    missing or malformed, returns an empty string instead of blindly slicing
    arbitrary characters. This helps avoid passing invalid values into Cypher
    date() expressions.
    """
    if not value:
        return ""

    raw = value.strip()
    if not raw:
        return ""

    # Prefer the date part before any time separator (e.g. "2024-01-01T12:00:00Z")
    date_part = raw.split("T", 1)[0]

    # Validate the input as ISO 8601. We try the full value first, then the
    # date component alone. If both fail, treat it as malformed.
    try:
        datetime.fromisoformat(raw)
    except ValueError:
        try:
            datetime.fromisoformat(date_part)
        except ValueError:
            logger.warning(
                "Invalid ISO date string passed to _safe_iso_date: %r", value)
            return ""

    return date_part[:10]


def _resolve_trend_window(state: AgentState) -> tuple[str, str, bool]:
    date_filter = state.get("date_filter")
    if date_filter:
        return date_filter[0], date_filter[1], False

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=settings.analytics_default_trend_days)
    return start.isoformat(), now.isoformat(), True


def _daily_activity_series(username: str, start_iso: str, end_iso: str) -> list[dict]:
    start_date = _safe_iso_date(start_iso)
    end_date = _safe_iso_date(end_iso)
    cypher = """
        WITH date($start_date) AS startDate, date($end_date) AS endDate
        WITH [d IN range(0, duration.inDays(startDate, endDate).days) |
              startDate + duration({days: d})] AS days
        UNWIND days AS day
        OPTIONAL MATCH (p:Post {user_context_username: $username})
        WHERE p.created_at IS NOT NULL
          AND p.created_at >= toString(day)
          AND p.created_at < toString(day + duration({days: 1}))
        WITH day, count(p) AS posts
        OPTIONAL MATCH (m:Message {user_context_username: $username})
        WHERE m.date IS NOT NULL
          AND size(toString(m.date)) >= 10
          AND date(substring(toString(m.date), 0, 10)) = day
        WITH day, posts, count(m) AS messages
        OPTIONAL MATCH (c:Comment {user_context_username: $username})
        WHERE c.time IS NOT NULL
          AND size(toString(c.time)) >= 10
          AND date(substring(toString(c.time), 0, 10)) = day
        WITH day, posts, messages, count(c) AS comments
        RETURN toString(day) AS day,
               posts,
               messages,
               comments,
               (posts + messages + comments) AS total
        ORDER BY day ASC
    """
    rows = _store.run_query(
        cypher,
        username=username,
        start_date=start_date,
        end_date=end_date,
    )

    return [
        {
            "day": str(r.get("day", "")),
            "posts": int(r.get("posts", 0)),
            "messages": int(r.get("messages", 0)),
            "comments": int(r.get("comments", 0)),
            "total": int(r.get("total", 0)),
        }
        for r in rows
    ]


def _bucket_activity_series(daily: list[dict], bucket_days: int) -> list[dict]:
    if bucket_days <= 1 or not daily:
        return daily

    bucketed: list[dict] = []
    for i in range(0, len(daily), bucket_days):
        chunk = daily[i:i + bucket_days]
        start = chunk[0]["day"]
        end = chunk[-1]["day"]
        bucketed.append(
            {
                "bucket_start": start,
                "bucket_end": end,
                "posts": sum(row["posts"] for row in chunk),
                "messages": sum(row["messages"] for row in chunk),
                "comments": sum(row["comments"] for row in chunk),
                "total": sum(row["total"] for row in chunk),
            }
        )
    return bucketed


def _build_trend_payload(state: AgentState, username: str) -> dict:
    start_iso, end_iso, defaulted_window = _resolve_trend_window(state)
    bucket_days = max(1, int(settings.analytics_bucket_days))
    daily = _daily_activity_series(username, start_iso, end_iso)
    buckets = _bucket_activity_series(daily, bucket_days)

    if not buckets:
        return {
            "kind": "trend",
            "query_type": "activity_time_series",
            "user_context": username,
            "time_window": {
                "start": start_iso,
                "end": end_iso,
                "bucket_days": bucket_days,
                "defaulted": defaulted_window,
            },
            "metrics": {
                "time_series": [],
                "totals": {"posts": 0, "messages": 0, "comments": 0, "total": 0},
            },
            "coverage": {
                "data_points": 0,
                "sparse": True,
                "source": "neo4j_graph",
            },
            "summary": f"No activity trend data found for user '{username}'.",
        }

    totals = {
        "posts": sum(int(x.get("posts", 0)) for x in daily),
        "messages": sum(int(x.get("messages", 0)) for x in daily),
        "comments": sum(int(x.get("comments", 0)) for x in daily),
    }
    totals["total"] = totals["posts"] + totals["messages"] + totals["comments"]

    first = buckets[0]["total"]
    last = buckets[-1]["total"]
    delta = last - first
    trend_direction = "stable"
    if delta > 0:
        trend_direction = "increasing"
    elif delta < 0:
        trend_direction = "decreasing"

    summary = (
        f"Activity trend for @{username} is {trend_direction} over "
        f"{start_iso[:10]} to {end_iso[:10]} "
        f"(delta {delta:+d} events per {bucket_days}-day bucket)."
    )

    return {
        "kind": "trend",
        "query_type": "activity_time_series",
        "user_context": username,
        "time_window": {
            "start": start_iso,
            "end": end_iso,
            "bucket_days": bucket_days,
            "defaulted": defaulted_window,
        },
        "metrics": {
            "time_series": buckets,
            "totals": totals,
            "trend": {
                "direction": trend_direction,
                "delta_first_to_last_bucket": delta,
            },
        },
        "coverage": {
            "data_points": len(buckets),
            "sparse": totals["total"] == 0,
            "source": "neo4j_graph",
        },
        "summary": summary,
    }


def _build_aggregate_payload(query: str, username: str) -> dict:
    top_limit = max(1, int(settings.analytics_top_entities))
    q = query.lower()

    if "message the most" in q:
        top = _top_message_partners(username, limit=top_limit)
        if not top:
            summary = f"No direct message history found for user '{username}'."
        else:
            best_name, best_count = top[0]
            summary = (
                f"You message {best_name} the most ({best_count} messages). "
                f"Top message partners: {_format_ranked(top)}."
            )
        return {
            "kind": "aggregate",
            "query_type": "top_message_partners",
            "user_context": username,
            "metrics": {"partners": [{"name": n, "count": c} for n, c in top]},
            "coverage": {"source": "neo4j_graph", "limit": top_limit},
            "summary": summary,
        }

    if "messages usually discuss" in q or "direct messages usually discuss" in q:
        top = _top_message_topics(username, limit=top_limit)
        if not top:
            summary = f"No recurring direct-message topics found for user '{username}'."
        else:
            summary = f"Your direct messages most often discuss: {_format_ranked(top)}."
        return {
            "kind": "aggregate",
            "query_type": "top_message_topics",
            "user_context": username,
            "metrics": {"topics": [{"name": n, "count": c} for n, c in top]},
            "coverage": {"source": "neo4j_graph", "limit": top_limit},
            "summary": summary,
        }

    if "engage with my posts" in q or "engages with my posts" in q:
        top = _top_engagers(username, limit=top_limit)
        if not top:
            summary = f"No engagement comments found for user '{username}'."
        else:
            best_name, best_count = top[0]
            summary = (
                f"{best_name} engages with your posts the most ({best_count} comments). "
                f"Top engagers: {_format_ranked(top)}."
            )
        return {
            "kind": "aggregate",
            "query_type": "top_engagers",
            "user_context": username,
            "metrics": {"engagers": [{"name": n, "count": c} for n, c in top]},
            "coverage": {"source": "neo4j_graph", "limit": top_limit},
            "summary": summary,
        }

    if "main topics i post" in q:
        top = _top_themes_authored(username, limit=top_limit)
        if not top:
            summary = f"No authored post themes found for user '{username}'."
        else:
            summary = f"Your main posting themes are: {_format_ranked(top)}."
        return {
            "kind": "aggregate",
            "query_type": "top_authored_themes",
            "user_context": username,
            "metrics": {"themes": [{"name": n, "count": c} for n, c in top]},
            "coverage": {"source": "neo4j_graph", "limit": top_limit},
            "summary": summary,
        }

    top = _top_themes_interactions(username, limit=top_limit)
    if not top:
        summary = f"I couldn't find enough analytics data for user '{username}'."
    else:
        summary = f"Top themes in your social graph are: {_format_ranked(top)}."
    return {
        "kind": "aggregate",
        "query_type": "top_interaction_themes",
        "user_context": username,
        "metrics": {"themes": [{"name": n, "count": c} for n, c in top]},
        "coverage": {"source": "neo4j_graph", "limit": top_limit},
        "summary": summary,
    }


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
    rows = _store.run_query(cypher, username=username, limit=limit)
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
    rows = _store.run_query(
        cypher,
        username=username,
        keywords=_MESSAGE_TOPIC_KEYWORDS,
        limit=limit,
    )
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
    rows = _store.run_query(cypher, username=username, limit=limit)
    return [(str(r.get("name", "")), int(r.get("c", 0))) for r in rows]


def _top_themes_authored(username: str, limit: int = 5) -> list[tuple[str, int]]:
    cypher = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_TAG]->(t:Tag)
        WHERE p.account_username = $username
        RETURN toLower(t.name) AS tag, count(*) AS c
        ORDER BY c DESC, tag ASC
        LIMIT $limit
    """
    rows = _store.run_query(cypher, username=username, limit=limit)
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
    rows = _store.run_query(cypher, username=username, limit=limit)

    ranked = [(str(r.get("tag", "")), int(r.get("c", 0))) for r in rows]
    if ranked:
        return ranked

    fallback = """
        MATCH (p:Post {user_context_username: $username})-[:HAS_TAG]->(t:Tag)
        RETURN toLower(t.name) AS tag, count(*) AS c
        ORDER BY c DESC, tag ASC
        LIMIT $limit
    """
    rows = _store.run_query(fallback, username=username, limit=limit)
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

        thread_hits = _store.similarity_search_threads(
            query_embedding=query_embedding,
            username=user_context,
            top_k=max(3, settings.default_top_k // 2),
        )
        results.extend(_to_thread_result(h) for h in thread_hits)

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
    if state.get("requires_graph_traversal"):
        logger.info(
            "Advanced retrieval guardrail redirected to analytics path.")
        return await analytics_retrieval_node(state)

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

            for h in _store.similarity_search_threads(
                query_embedding=emb,
                username=user_context,
                top_k=3,
            ):
                results.append(_to_thread_result(h))

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
        neighbors = _store.graph_neighbors(r.post.id)
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
    """Deterministic aggregate/trend analytics over user-scoped graph data."""
    query = (state.get("query") or "").lower()
    user_context = _extract_user_context(state)
    analytics_kind = state.get("analytics_kind", "none")

    if not user_context:
        payload = {
            "kind": analytics_kind if analytics_kind != "none" else "aggregate",
            "query_type": "missing_user_context",
            "user_context": "",
            "metrics": {},
            "coverage": {"source": "neo4j_graph", "data_points": 0},
            "summary": (
                "I can run analytics once you provide a user context, "
                "for example: 'for albert336'."
            ),
        }
        return {
            "search_results": [],
            "analytics_payload": payload,
            "answer": payload["summary"],
        }

    is_trend_query = analytics_kind == "trend" or state.get(
        "intent") == "trend_analysis"
    if is_trend_query:
        payload = _build_trend_payload(state, user_context)
    else:
        payload = _build_aggregate_payload(query, user_context)

    return {
        "search_results": [],
        "analytics_payload": payload,
        "answer": payload.get("summary", ""),
    }
