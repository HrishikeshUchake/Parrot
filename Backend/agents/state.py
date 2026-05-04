"""LangGraph shared state definition."""
from __future__ import annotations
from typing import TypedDict, Annotated, NotRequired
import operator

from ..database.models import SearchResult


class AgentState(TypedDict):
    # ── Input
    query: str
    user_context_username: NotRequired[str]
    llm_mode: NotRequired[str]              # "remote" | "local"
    debug_pipeline: NotRequired[bool]
    session_id: NotRequired[str]
    # ── After QueryAnalyzer
    intent: str
    entities: list[str]
    filters: dict
    date_filter: NotRequired[tuple[str, str] | None]
    tag_filter: NotRequired[list[str] | None]
    sub_queries: list[str]
    complexity: str
    requires_graph_traversal: NotRequired[bool]
    analytics_kind: NotRequired[str]          # "none" | "aggregate" | "trend"
    aggregate_query_type: NotRequired[str]    # "none" | "top_message_partners" | "top_message_topics" | "top_engagers" | "top_authored_themes" | "top_interaction_themes"

    # ── After Router
    route: str                              # "simple" | "analytics" | "advanced"

    # ── After Retrieval (reducer accumulates across branches)
    search_results: Annotated[list[SearchResult], operator.add]
    analytics_payload: NotRequired[dict]
    cache_hit: NotRequired[bool]
    cache_hit_type: NotRequired[str]
    cache_similarity: NotRequired[float]

    # ── After Synthesis
    answer: str
    reasoning: str
    privacy_debug: NotRequired[dict]

    # ── Error channel
    error: str
