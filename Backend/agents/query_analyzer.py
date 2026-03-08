"""QueryAnalyzer node – classifies intent and extracts entities."""
from __future__ import annotations
import json
import logging

from .state import AgentState
from ..llm.ollama_client import OllamaClient
from ..llm.prompts import QUERY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)
_client = OllamaClient()


async def query_analyzer_node(state: AgentState) -> dict:
    """Classify the user query and extract structured metadata."""
    query = state["query"]
    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)

    try:
        raw = await _client.generate(prompt)
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

    return {
        "intent": analysis.get("intent", "open_ended"),
        "entities": analysis.get("entities", []),
        "filters": analysis.get("filters", {}),
        "sub_queries": analysis.get("sub_queries", []),
        "complexity": analysis.get("complexity", "simple"),
    }
