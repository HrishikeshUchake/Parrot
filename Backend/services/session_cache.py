from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import settings
from ..database.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class CacheLookupResult:
    search_results: list[SearchResult]
    analytics_payload: dict[str, Any] | None
    hit_type: str
    similarity: float | None = None


@dataclass
class CacheEntry:
    query: str
    normalized_query: str
    structured_key: str
    query_embedding: list[float] | None
    user_context_username: str
    route: str
    intent: str
    entities: list[str] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    analytics_payload: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionRetrievalCache:
    """
    In-memory retrieval cache grouped by chat/session.

    Important:
    - This does NOT persist to disk.
    - This does NOT cache final answers.
    - This only caches retrieved context/results for the current app process.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[CacheEntry]] = {}

    def get(
        self,
        *,
        session_id: str,
        query: str,
        structured_key: str,
        entities: list[str],
        query_embedding: list[float] | None = None,
    ) -> CacheLookupResult | None:
        if not settings.session_cache_enabled:
            return None

        entries = self._sessions.get(session_id, [])
        if not entries:
            logger.info(
                "\n\n[CACHE]\n"
                "  Status: MISS\n"
                "  Reason: no_entries\n"
                "  Session: %s\n",
                session_id,
            )
            return None

        normalized = normalize_query(query)

        # 1. Exact/normalized hit
        for entry in reversed(entries):
            if entry.normalized_query == normalized:
                logger.info(
                    "\n\n[CACHE]\n"
                    "  Status: HIT\n"
                    "  Type: normalized\n"
                    "  Session: %s\n"
                    "  Query: %s\n",
                    session_id,
                    query,
                )
                return CacheLookupResult(
                    search_results=entry.search_results,
                    analytics_payload=entry.analytics_payload,
                    hit_type="normalized",
                )

        # 2. Structured retrieval-plan hit
        for entry in reversed(entries):
            if entry.structured_key == structured_key:
                logger.info(
                    "\n\n[CACHE]\n"
                    "  Status: HIT\n"
                    "  Type: structured\n"
                    "  Session: %s\n"
                    "  Query: %s\n",
                    session_id,
                    query,
                )
                return CacheLookupResult(
                    search_results=entry.search_results,
                    analytics_payload=entry.analytics_payload,
                    hit_type="structured",
                )

        # 3. Semantic hit
        if query_embedding is not None:
            best_entry: CacheEntry | None = None
            best_similarity = -1.0
            current_entities = entities

            for entry in entries:
                if entry.query_embedding is None:
                    continue

                if not entities_compatible(current_entities, entry.entities):
                    continue

                similarity = cosine_similarity(query_embedding, entry.query_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if (
                best_entry is not None
                and best_similarity >= settings.session_cache_semantic_threshold
            ):
                logger.info(
                    "\n\n[CACHE]\n"
                    "  Status: HIT\n"
                    "  Type: semantic\n"
                    "  Similarity: %.3f\n"
                    "  Session: %s\n"
                    "  Query: %s\n",
                    best_similarity,
                    session_id,
                    query,
                )
                return CacheLookupResult(
                    search_results=best_entry.search_results,
                    analytics_payload=best_entry.analytics_payload,
                    hit_type="semantic",
                    similarity=best_similarity,
                )

        logger.info(
            "\n\n[CACHE]\n"
            "  Status: MISS\n"
            "  Session: %s\n"
            "  Query: %s\n",
            session_id,
            query,
        )
        return None
    
    def get_multi_entity(
        self,
        *,
        session_id: str,
        query: str,
        entities: list[str],
    ) -> CacheLookupResult | None:
        if not settings.session_cache_enabled:
            return None

        # Only trigger if multiple entities
        if len(entities) < 2:
            return None

        entries = self._sessions.get(session_id, [])
        if not entries:
            return None

        matched_entries: list[CacheEntry] = []

        for entity in entities:
            best_match: CacheEntry | None = None

            # find most recent matching entry for each entity
            for entry in reversed(entries):
                if entity in entry.entities:
                    if entry.search_results:
                        best_match = entry
                        break

            if best_match is None:
                return None  # require all entities

            matched_entries.append(best_match)

        # merge results
        merged_results: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()

        for entry in matched_entries:
            for result in entry.search_results:
                key = (result.result_type, result.item_id)
                if key in seen:
                    continue
                seen.add(key)
                merged_results.append(result)

        logger.info(
            "\n\n[CACHE]\n"
            "  Status: HIT\n"
            "  Type: multi_entity\n"
            "  Session: %s\n"
            "  Query: %s\n"
            "  Entities: %s\n"
            "  Combined Entries: %d\n"
            "  Combined Results: %d\n",
            session_id,
            query,
            entities,
            len(matched_entries),
            len(merged_results),
        )

        return CacheLookupResult(
            search_results=merged_results,
            analytics_payload=None,
            hit_type="multi_entity",
        )

    def set(
        self,
        *,
        session_id: str,
        query: str,
        structured_key: str,
        query_embedding: list[float] | None,
        user_context_username: str,
        route: str,
        intent: str,
        entities: list[str],
        search_results: list[SearchResult],
        analytics_payload: dict[str, Any] | None = None,
    ) -> None:
        if not settings.session_cache_enabled:
            return

        entry = CacheEntry(
            query=query,
            normalized_query=normalize_query(query),
            structured_key=structured_key,
            query_embedding=query_embedding,
            user_context_username=user_context_username,
            route=route,
            intent=intent,
            entities=entities,
            search_results=search_results,
            analytics_payload=analytics_payload,
        )

        session_entries = self._sessions.setdefault(session_id, [])
        session_entries.append(entry)

        max_entries = max(1, int(settings.session_cache_max_entries))
        if len(session_entries) > max_entries:
            del session_entries[:-max_entries]

        logger.info(
            "\n\n[CACHE]\n"
            "  Status: SAVED\n"
            "  Session: %s\n"
            "  Route: %s\n"
            "  Intent: %s\n"
            "  Entities: %s\n"
            "  Results Count: %d\n"
            "  Analytics Payload: %s\n",
            session_id,
            route,
            intent,
            entities,
            len(search_results),
            analytics_payload is not None,
        )

_ENTITY_STOP_WORDS = {
    "i", "me", "my", "mine", "self", "you", "your",
    "messages", "message", "posts", "post", "stories", "story",
    "conversation", "conversations", "chat", "chats", "convo", "convos",
    "people", "contacts", "top contacts", "anything",
}


def normalize_entities(raw_entities: list[Any]) -> list[str]:
    cleaned: list[str] = []

    for raw in raw_entities or []:
        entity = str(raw).lower().strip()
        entity = entity.replace("*", "")
        entity = entity.lstrip("@").strip()

        if not entity:
            continue

        if entity in _ENTITY_STOP_WORDS:
            continue

        cleaned.append(entity)

    return sorted(set(cleaned))


def entities_compatible(current_entities: list[str], cached_entities: list[str]) -> bool:
    current = set(current_entities)
    cached = set(cached_entities)

    if current:
        return current.issubset(cached) or current == cached

    return not cached


def normalize_query(query: str) -> str:
    """
    Normalize a query so small formatting differences still match.

    Example:
    'Who do I message the most?!'
    becomes:
    'who do i message the most'
    """
    text = (query or "").lower().strip()
    text = re.sub(r"[^a-z0-9_@\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_structured_cache_key(state: dict[str, Any], user_context_username: str) -> str:
    """
    Build a safe cache key based on the analyzed retrieval plan.

    This prevents bad reuse across different users, routes, entities, filters,
    or analytics query types.
    """
    entities = sorted(
        str(e).lower().lstrip("@").strip()
        for e in state.get("entities", [])
        if str(e).strip()
    )

    filters = state.get("filters") or {}
    normalized_filters = {
        str(k): filters[k]
        for k in sorted(filters.keys())
    }

    parts = {
        "user": user_context_username or "",
        "route": state.get("route", ""),
        "intent": state.get("intent", ""),
        "entities": entities,
        "filters": normalized_filters,
        "date_filter": state.get("date_filter"),
        "tag_filter": state.get("tag_filter"),
        "analytics_kind": state.get("analytics_kind", "none"),
        "aggregate_query_type": state.get("aggregate_query_type", "none"),
    }

    return repr(parts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity.

    Your embeddings are already normalized in EmbeddingService, so this is
    usually equivalent to a dot product. This function still handles general
    vectors safely.
    """
    if not a or not b or len(a) != len(b):
        return -1.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0

    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


session_retrieval_cache = SessionRetrievalCache()