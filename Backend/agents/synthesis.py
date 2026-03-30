"""Synthesis node generates final answer with citations using LLM + Privacy."""
from __future__ import annotations
import logging

from .state import AgentState
from ..database.models import SearchResult

# from ..llm.ollama_client import OllamaClient
# from ..llm.openrouter_client import OpenRouterClient
# =======
from ..llm.llm_provider import get_llm_provider

from ..llm.prompts import SYNTHESIS_PROMPT
from ..services.privacy import PresidioPrivatizer, NoOpPrivatizer
from ..config import settings

logger = logging.getLogger(__name__)


# Initialize LLM client based on configuration
# def _init_llm_client():
#     """Initialize the appropriate LLM client (Ollama or OpenRouter)."""
#     if settings.llm_provider.lower() == "openrouter":
#         return OpenRouterClient()
#     else:
#         return OllamaClient()


# Initialize privacy layer
def _init_privatizer():
    """Initialize the appropriate privatizer (Presidio or NoOp)."""
    if not settings.privacy_enabled:
        return NoOpPrivatizer()

    if settings.privacy_anonymizer.lower() == "presidio":
        try:
            return PresidioPrivatizer()
        except ImportError:
            logger.warning("Presidio not available, using NoOp privatizer")
            return NoOpPrivatizer()
    else:
        return NoOpPrivatizer()


_privatizer = _init_privatizer()
_client = get_llm_provider()


def _post_label(p) -> str:
    """Return a short human-readable citation label for a post."""
    author = p.account_acct or p.account_username or "unknown"
    # Take up to 80 chars of content, ending on a word boundary
    text = p.content.strip()
    if len(text) > 80:
        cut = text[:80].rsplit(None, 1)[0]
        summary = cut + "…"
    else:
        summary = text
    return f'"{summary}" (@{author})'


def _result_label(r: SearchResult) -> str:
    if r.result_type == "post" and r.post is not None:
        return _post_label(r.post)
    if r.result_type == "message":
        sender = r.metadata.get("sender_name", "unknown")
        receiver = r.metadata.get("receiver_name", "unknown")
        text = (r.content or "").strip()
        text = (text[:80].rsplit(None, 1)[0] +
                "...") if len(text) > 80 else text
        return f'Message "{text}" ({sender} -> {receiver})'
    if r.result_type == "comment":
        commenter = r.metadata.get("commenter_name", "unknown")
        post_id = r.metadata.get("post_id", "")
        text = (r.content or "").strip()
        text = (text[:80].rsplit(None, 1)[0] +
                "...") if len(text) > 80 else text
        return f'Comment "{text}" by @{commenter} on Post {post_id}'
    return f'{r.result_type} #{r.item_id}'



def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant context found."
    lines = []
    for r in results:
        if r.result_type == "post" and r.post is not None:
            p = r.post
            snippet = p.content[:400] + ("..." if len(p.content) > 400 else "")
            author = p.account_acct or p.account_username or "unknown"
            engagement = (
                f"reblogs={p.reblogs_count}  favs={p.favourites_count}  "
                f"replies={p.replies_count}"
            )
            label = _result_label(r)
            lines.append(
                f"[{label}] Score={r.score:.2f} | type=post | @{author} | {engagement}\n"
                f"Tags={p.tags}\n"
                f"Content: {snippet}"
            )
            continue

        snippet = r.content[:400] + ("..." if len(r.content) > 400 else "")
        lines.append(
            f"[{_result_label(r)}] Score={r.score:.2f} | type={r.result_type}\n"
            f"Metadata={r.metadata}\n"
            f"Content: {snippet}"
        )
    return "\n---\n".join(lines)


async def synthesis_node(state: AgentState) -> dict:
    """Generate a final answer grounded in retrieved documents with privacy protection.

    Flow:
    1. Format context from search results
    2. Anonymize content using Presidio (if enabled)
    3. Send anonymized context to LLM
    4. Restore tokens in response
    5. Return restored answer to user
    """
    results = state.get("search_results", [])
    context = _format_context(results)

    # Step 1: Anonymize before sending to LLM
    anonymized_context = _privatizer.privatize_context(context)

    logger.debug(f"Original context length: {len(context)}")
    logger.debug(f"Anonymized context length: {len(anonymized_context)}")
    if settings.privacy_enabled:
        logger.debug(f"PII mappings: {_privatizer.get_current_mappings()}")

    # Step 2: Create prompt with anonymized context
    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        context=anonymized_context
    )

    try:
        # Step 3: Call LLM with anonymized content
        answer = await _client.generate(prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        answer = (
            "I was unable to generate a response at this time. "
            f"Found {len(results)} relevant posts."
        )
        return {
            "answer": answer,
            "reasoning": f"Route: {state.get('route', 'unknown')} | Results: {len(results)} | Error: {str(exc)}",
        }

    # Step 4: Restore tokens in the answer
    restored_answer = _privatizer.restore(answer)

    return {
        "answer": restored_answer,
        "reasoning": f"Route: {state.get('route', 'unknown')} | Results: {len(results)}",
    }
