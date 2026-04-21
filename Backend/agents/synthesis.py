"""Synthesis node generates final answer with citations using Llama."""
from __future__ import annotations
import logging

from .state import AgentState
from ..database.models import SearchResult
from ..llm.llm_provider import get_node_llm_provider
from ..llm.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)
_client = get_node_llm_provider("synthesis")


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
    if r.result_type == "thread" and r.thread is not None:
        parts = ", ".join(r.thread.participants)
        return f'Conversation thread involving {parts}'
    return f'{r.result_type} #{r.item_id}'

def _result_brief(r: SearchResult) -> str:
    return f"type={r.result_type} | id={r.item_id} | score={r.score:.3f}"

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

        if r.result_type == "thread" and r.thread is not None:
            t = r.thread
            # Combine the overall summary with the detailed raw messages for full context
            snippet = f"--- Thread Summary ---\n{t.summary}\n\n--- Thread Messages ---\n"
            for m in t.messages:
                author_name = m.get('author', m.get('sender', 'unknown'))
                time_str = m.get('time', '')
                content_str = m.get('content', m.get('text', ''))
                snippet += f" [{time_str}] {author_name}: {content_str}\n"
            # Threads can be quite large, allow larger snippet context
            snippet = snippet[:1500] + ("..." if len(snippet) > 1500 else "")
            lines.append(
                f"[{_result_label(r)}] Score={r.score:.2f} | type=thread\n"
                f"Participants={t.participants}\n"
                f"Context:\n{snippet}"
            )
            continue

        snippet = r.content[:400] + ("..." if len(r.content) > 400 else "")
        lines.append(
            f"[{_result_label(r)}] Score={r.score:.2f} | type={r.result_type}\n"
            f"Metadata={r.metadata}\n"
            f"Content: {snippet}"
        )
    return "\n---\n".join(lines)


def _render_analytics_answer(payload: dict) -> str:
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        summary = "Analytics computed from graph traversal."

    kind = payload.get("kind", "aggregate")
    metrics = payload.get("metrics", {})
    lines = [summary]

    if kind == "trend":
        window = payload.get("time_window", {})
        start = window.get("start", "")
        end = window.get("end", "")
        bucket_days = window.get("bucket_days", "")
        if start and end:
            lines.append(
                f"Window: {start[:10]} to {end[:10]} (bucket={bucket_days}d)")

        totals = metrics.get("totals", {})
        if totals:
            lines.append(
                "Totals: "
                f"posts={totals.get('posts', 0)}, "
                f"messages={totals.get('messages', 0)}, "
                f"comments={totals.get('comments', 0)}, "
                f"all_activity={totals.get('total', 0)}"
            )

        series = metrics.get("time_series", [])
        if series:
            preview = series[-3:]
            preview_text = ", ".join(
                f"{row.get('bucket_start', row.get('day', ''))}->{row.get('bucket_end', row.get('day', ''))}: {row.get('total', 0)}"
                for row in preview
            )
            lines.append(f"Recent buckets: {preview_text}")
    else:
        for key in ("partners", "topics", "engagers", "themes"):
            rows = metrics.get(key)
            if not rows:
                continue
            preview = ", ".join(
                f"{item.get('name', '')} ({item.get('count', 0)})" for item in rows[:5]
            )
            lines.append(f"Top {key}: {preview}")

    return "\n".join(lines)


def _with_user_perspective_context(context: str, username: str | None) -> str:
    """Add user-perspective guidance to synthesis context when username is available."""
    if not username:
        return context
    preface = (
        "Assume you are answering on behalf of the user or analyzing the data "
        "for the user. The primary user asking the question is '@"
        f"{username}'. When referring to 'my' or 'I' in the query, it means "
        f"@{username}."
    )
    return f"{preface}\n\n{context}"


async def synthesis_node(state: AgentState) -> dict:
    """Generate a final answer grounded in retrieved documents."""
    analytics_payload = state.get("analytics_payload")
    if analytics_payload:
        context = _render_analytics_answer(analytics_payload)
        context = _with_user_perspective_context(context, state.get("user_context_username"))

        logger.info(
            "\n\n[SYNTHESIS_ANALYTICS]\n"
            "  Route: %s\n"
            "  Query: %s\n"
            "  Payload Kind: %s\n"
            "  Payload Query Type: %s\n",
            state.get("route", "unknown"),
            state["query"],
            analytics_payload.get("kind"),
            analytics_payload.get("query_type"),
        )

        prompt = SYNTHESIS_PROMPT.format(query=state["query"], context=context)
        try:
            answer = await _client.generate(prompt)
        except Exception as exc:
            logger.error("Synthesis LLM call failed: %s", exc)
            answer = analytics_payload.get("summary", "Analytics computed from graph.")
        return {
            "answer": answer,
            "reasoning": (
                f"Route: {state.get('route', 'unknown')} | "
                "Source: graph_analytics_payload"
            ),
        }

    # Preserve deterministic retrieval direct answers (e.g. meta/count paths).
    # Retrieval sets `answer` directly and can intentionally return no sources.
    # In that case, avoid LLM synthesis overwriting a known-correct answer.
    existing_answer = (state.get("answer") or "").strip()
    if existing_answer and not state.get("search_results"):
        return {
            "answer": existing_answer,
            "reasoning": (
                f"Route: {state.get('route', 'unknown')} | "
                "Source: retrieval_direct_answer"
            ),
        }

    results = state.get("search_results", [])
    context = _format_context(results)

    context = _with_user_perspective_context(
        context,
        state.get("user_context_username"),
    )

    logger.info(
        "\n\n[SYNTHESIS]\n"
        "  Route: %s\n"
        "  Query: %s\n"
        "  Num Results: %d\n",
        state.get("route", "unknown"),
        state["query"],
        len(results),
    )
    for r in results:
        logger.info("  %s", _result_brief(r))

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
