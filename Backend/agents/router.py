"""Router node decides retrieval strategy based on query analysis."""
from __future__ import annotations
import logging

from .state import AgentState
from ..llm.llm_provider import get_llm_provider
from ..llm.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)
_client = get_llm_provider()


async def router_node(state: AgentState) -> dict:
    """Decide between simple and advanced retrieval."""

    # Use structured analysis as primary signal
    complexity = state.get("complexity", "simple")
    intent = state.get("intent", "open_ended")
    sub_queries = state.get("sub_queries", [])

    ADVANCED_INTENTS = {"trend_analysis", "comparison", "open_ended", "summary"}

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
            decision = await _client.generate(prompt)
            route = "advanced" if "advanced" in decision.lower() else "simple"
        except Exception as exc:
            logger.warning(
                "Router LLM call failed: %s defaulting simple", exc)
            route = "simple"

    logger.info("Router decision: %s (intent=%s, complexity=%s)",
                route, intent, complexity)
    return {"route": route}


def route_decision(state: AgentState) -> str:
    """Conditional edge function for LangGraph."""
    return state.get("route", "simple")
