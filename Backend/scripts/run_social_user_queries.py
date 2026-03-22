from __future__ import annotations

import asyncio
from pprint import pprint

from Backend.agents.graph import rag_graph

USER = "albert336"
QUERY_TIMEOUT_SECONDS = 45
QUERIES = [
    "How many posts are in my data?",
    "What are the main topics I post about?",
    "Who do I message the most?",
    "What do my direct messages usually discuss?",
    "Which people engage with my posts the most?",
    "What themes appear most in posts I interact with?",
    "What were my recent comments talking about?",
    "How was my tone in my previous conversations with nina_q?",
    "What do I usually discuss with nina_q in DMs?",
    "Are my conversations with nina_q mostly positive or neutral?",
    "Summarize my last few interactions with nina_q.",
    "Which people do I message the most besides sam_lee?",
    "What topics come up most in my chats with nina_q?",
    "Which themes appear in posts where I left comments recently?",
    "Who comments most on my posts about health and fitness?",
    "What are my recent posts mostly about?",
]


async def main() -> None:
    out = []
    for q in QUERIES:
        state = {"query": q, "search_results": [], "user_context_username": USER}
        try:
            result = await asyncio.wait_for(
                rag_graph.ainvoke(state), timeout=QUERY_TIMEOUT_SECONDS
            )
            out.append(
                {
                    "query": q,
                    "route": result.get("route", ""),
                    "num_sources": len(result.get("search_results", [])),
                    "answer": result.get("answer", "").strip(),
                    "reasoning": result.get("reasoning", ""),
                }
            )
        except TimeoutError:
            out.append(
                {
                    "query": q,
                    "route": "",
                    "num_sources": 0,
                    "answer": "",
                    "reasoning": f"timeout_after_{QUERY_TIMEOUT_SECONDS}s",
                    "error": "query_timed_out",
                }
            )
        except Exception as exc:
            out.append(
                {
                    "query": q,
                    "route": "",
                    "num_sources": 0,
                    "answer": "",
                    "reasoning": "query_failed",
                    "error": str(exc),
                }
            )
    pprint(out)


if __name__ == "__main__":
    asyncio.run(main())
