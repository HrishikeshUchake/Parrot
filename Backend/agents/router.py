"""Router node decides retrieval strategy based on query analysis."""
from __future__ import annotations
import logging

from .state import AgentState
from ..llm.llm_provider import get_node_llm_provider, get_llm_provider_for_backend
from ..llm.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)
_client = get_node_llm_provider("router")


def _resolve_llm_client(state: AgentState):
    mode = str(state.get("llm_mode", "")).strip().lower()
    if mode == "local":
        return get_llm_provider_for_backend("ollama")
    if mode == "remote":
        return get_llm_provider_for_backend("openai")
    return _client


async def router_node(state: AgentState) -> dict:
    """Decide between simple and advanced retrieval."""

    # Use structured analysis as primary signal
    complexity = state.get("complexity", "simple")
    intent = state.get("intent", "open_ended")
    sub_queries = state.get("sub_queries", [])
    requires_graph = bool(state.get("requires_graph_traversal", False))
    analytics_kind = state.get("analytics_kind", "none")

    # Hardcode 'identity' and 'meta' to always map to the 'simple' route
    if intent in {"identity", "meta"}:
        return {"route": "simple"}

    ADVANCED_INTENTS = {"trend_analysis",
                        "comparison", "open_ended", "summary"}
    llm_client = _resolve_llm_client(state)

    if requires_graph:
        route = "analytics"
        logger.info(
            "\n\n[ROUTER]\n"
            "  Route: %s\n"
            "  Reason: requires_graph_traversal\n"
            "  Intent: %s\n"
            "  Complexity: %s\n"
            "  Requires Graph: %s\n"
            "  Analytics Kind: %s\n"
            "  Sub Queries: %s\n",
            route,
            intent,
            complexity,
            requires_graph,
            analytics_kind,
            sub_queries,
        )
        return {"route": route}

    if intent == "analytics":
        route = "analytics"
        logger.info(
            "\n\n[ROUTER]\n"
            "  Route: %s\n"
            "  Reason: intent_is_analytics\n"
            "  Intent: %s\n"
            "  Complexity: %s\n"
            "  Requires Graph: %s\n"
            "  Analytics Kind: %s\n"
            "  Sub Queries: %s\n",
            route,
            intent,
            complexity,
            requires_graph,
            analytics_kind,
            sub_queries,
        )
        return {"route": route}

    # Deterministic routing based on query analysis
    if complexity == "complex" or intent in ADVANCED_INTENTS or len(sub_queries) > 0:
        route = "advanced"
    else:
        # Use LLM only for ambiguous simple cases
        analysis_summary = (
            f"intent={intent}, "
            f"complexity={complexity}, "
            f"entities={state.get('entities')}, "
            f"sub_queries={sub_queries}"
        )
        prompt = ROUTER_PROMPT.format(analysis=analysis_summary)
        try:
            decision = await llm_client.generate(prompt)
            route = "advanced" if "advanced" in decision.lower() else "simple"
        except Exception as exc:
            logger.warning(
                "Router LLM call failed: %s defaulting simple", exc)
            route = "simple"

    logger.info(
        "\n\n[ROUTER]\n"
        "  Route: %s\n"
        "  Reason: standard_route_decision\n"
        "  Intent: %s\n"
        "  Complexity: %s\n"
        "  Requires Graph: %s\n"
        "  Analytics Kind: %s\n"
        "  Sub Queries: %s\n",
        route,
        intent,
        complexity,
        requires_graph,
        analytics_kind,
        sub_queries,
    )
    return {"route": route}


def route_decision(state: AgentState) -> str:
    """Conditional edge function for LangGraph."""
    return state.get("route", "simple")
