"""LangGraph shared state definition."""
from __future__ import annotations
from typing import TypedDict, Annotated, NotRequired
import operator

from ..database.models import SearchResult


class AgentState(TypedDict):
    # ── Input
    query: str
    user_context_username: NotRequired[str]

    # ── After QueryAnalyzer
    intent: str
    entities: list[str]
    filters: dict
    sub_queries: list[str]
    complexity: str

    # ── After Router
    route: str                              # "simple" | "advanced"

    # ── After Retrieval (reducer accumulates across branches)
    search_results: Annotated[list[SearchResult], operator.add]

    # ── After Synthesis
    answer: str
    reasoning: str

    # ── Error channel
    error: str
