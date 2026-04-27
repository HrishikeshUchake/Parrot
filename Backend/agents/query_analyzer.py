"""QueryAnalyzer node – classifies intent and extracts entities."""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from .state import AgentState
from ..llm.llm_provider import get_node_llm_provider, get_llm_provider_for_backend
from ..llm.prompts import QUERY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

def _truncate(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."

_client = get_node_llm_provider("query_analyzer")


def _resolve_llm_client(state: AgentState):
    mode = str(state.get("llm_mode", "")).strip().lower()
    if mode == "local":
        return get_llm_provider_for_backend("ollama")
    if mode == "remote":
        return get_llm_provider_for_backend("openai")
    return _client


def _requires_graph_traversal(intent: str, analytics_kind: str) -> bool:
    return intent in {"analytics", "trend_analysis"} or analytics_kind in {"aggregate", "trend"}


def _parse_date_range(date_str: str) -> tuple[str, str] | None:
    """Parse natural language date strings into ISO-8601 tuple (start, end)."""
    if not date_str:
        return None

    date_str = date_str.lower().strip()
    now = datetime.now(timezone.utc)

    try:
        if date_str == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start.isoformat(), now.isoformat())
        elif date_str in ("yesterday", "last 24 hours", "last 24h"):
            start = now - timedelta(days=1)
            return (start.isoformat(), now.isoformat())
        elif date_str in ("last week", "past week", "last 7 days"):
            start = now - timedelta(days=7)
            return (start.isoformat(), now.isoformat())
        elif date_str in ("last month", "past month", "last 30 days"):
            start = now - timedelta(days=30)
            return (start.isoformat(), now.isoformat())
        elif date_str == "this year":
            start = now.replace(month=1, day=1, hour=0,
                                minute=0, second=0, microsecond=0)
            return (start.isoformat(), now.isoformat())

        # Basic regex for YYYY-MM-DD to YYYY-MM-DD
        m = re.match(
            r"(\d{4}-\d{2}-\d{2})\s*(?:to|and|-)\s*(\d{4}-\d{2}-\d{2})", date_str)
        if m:
            start_dt = datetime.strptime(
                m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(m.group(
                2), "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            return (start_dt.isoformat(), end_dt.isoformat())

        # Add LLM parsing call if it gets complex, but for now we fallback gracefully
        # If parser fails, skip filter gracefully.
        return None
    except Exception as e:
        logger.warning(f"Failed to parse date range '{date_str}': {e}")
        return None


async def query_analyzer_node(state: AgentState) -> dict:
    """Classify the user query and extract structured metadata."""
    query = state["query"]
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
    llm_client = _resolve_llm_client(state)

    try:
        raw = await llm_client.generate(prompt)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        analysis = json.loads(raw)
    except Exception as exc:
        logger.warning(
            "QueryAnalyzer LLM call failed: %s using defaults", exc)
        analysis = {
            "intent": "open_ended",
            "entities": [],
            "filters": {},
            "sub_queries": [],
            "complexity": "simple",
        }

    filters = analysis.get("filters", {})
    tags = filters.get("tags", [])
    raw_date = filters.get("date_range", None)

    date_filter = _parse_date_range(raw_date) if raw_date else None

    intent = analysis.get("intent", "open_ended")
    complexity = analysis.get("complexity", "simple")
    sub_queries = analysis.get("sub_queries", [])
    analytics_kind = analysis.get("analytics_kind", "none")
    aggregate_query_type = analysis.get("aggregate_query_type", "none")

    is_analytics = intent == "analytics" or analytics_kind in {"aggregate", "trend"}
    is_trend = intent == "trend_analysis" or analytics_kind == "trend"
    requires_graph_traversal = _requires_graph_traversal(intent, analytics_kind)

    if is_trend:
        analytics_kind = "trend"
        complexity = "complex"
        if not sub_queries:
            sub_queries = [query]
    elif is_analytics:
        analytics_kind = "aggregate"
        complexity = "complex"
        if not sub_queries:
            sub_queries = [query]

    if requires_graph_traversal and not sub_queries:
        sub_queries = [query]

    logger.info(
        "\n\n[QUERY_ANALYZER]\n"
        "  Query: %s\n"
        "  Intent: %s\n"
        "  Entities: %s\n"
        "  Filters: %s\n"
        "  Date Filter: %s\n"
        "  Sub Queries: %s\n"
        "  Complexity: %s\n"
        "  Requires Graph: %s\n"
        "  Analytics Kind: %s\n"
        "  Aggregate Type: %s\n",
        _truncate(query),
        intent,
        analysis.get("entities", []),
        filters,
        date_filter,
        sub_queries,
        complexity,
        requires_graph_traversal,
        analytics_kind,
        aggregate_query_type,
    )

    return {
        "intent": intent,
        "entities": analysis.get("entities", []),
        "filters": filters,
        "date_filter": date_filter,
        "tag_filter": tags if tags else None,
        "sub_queries": sub_queries,
        "complexity": complexity,
        "requires_graph_traversal": requires_graph_traversal,
        "analytics_kind": analytics_kind,
        "aggregate_query_type": aggregate_query_type,
    }
