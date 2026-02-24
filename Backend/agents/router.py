"""Router node – decides retrieval strategy based on query analysis."""
from __future__ import annotations
import logging

from .state import AgentState
from ..llm.ollama_client import OllamaClient
from ..llm.prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)
_client = OllamaClient()


async def router_node(state: AgentState) -> dict:
    """Decide between simple and advanced retrieval."""
    analysis_summary = (
        f"intent={state.get('intent')}, "
        f"complexity={state.get('complexity')}, "
        f"entities={state.get('entities')}, "
        f"sub_queries={state.get('sub_queries')}"
    )
    prompt = ROUTER_PROMPT.format(analysis=analysis_summary)

    try:
        decision = await _client.generate(prompt)
        route = "advanced" if "advanced" in decision.lower() else "simple"
    except Exception as exc:
        logger.warning("Router LLM call failed: %s – defaulting by heuristic", exc)
        route = "advanced" if state.get("complexity") == "complex" else "simple"

    logger.info("Router decision: %s", route)
    return {"route": route}


def route_decision(state: AgentState) -> str:
    """Conditional edge function for LangGraph."""
    return state.get("route", "simple")
