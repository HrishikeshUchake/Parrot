"""Synthesis node – generates final answer with citations using Llama."""
from __future__ import annotations
import logging

from .state import AgentState
from ..database.models import SearchResult
from ..llm.ollama_client import OllamaClient
from ..llm.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
_client = OllamaClient()


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant posts found."
    lines = []
    for r in results:
        p = r.post
        body_snippet = p.body[:400] + ("..." if len(p.body) > 400 else "")
        lines.append(
            f"[Post #{p.id}] Score={r.score:.2f} | Tags={p.tags}\n"
            f"Title: {p.title}\n"
            f"Body: {body_snippet}"
        )
    return "\n---\n".join(lines)


async def synthesis_node(state: AgentState) -> dict:
    """Generate a final answer grounded in retrieved documents."""
    results = state.get("search_results", [])
    context = _format_context(results)
    prompt = SYNTHESIS_PROMPT.format(query=state["query"], context=context)

    try:
        answer = await _client.generate(prompt)
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        answer = (
            "I was unable to generate a response at this time. "
            f"Found {len(results)} relevant posts."
        )

    return {
        "answer": answer,
        "reasoning": f"Route: {state.get('route', 'unknown')} | Results: {len(results)}",
    }
