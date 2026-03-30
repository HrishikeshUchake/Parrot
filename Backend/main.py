"""
Parrot Agentic RAG System for Social Media Analysis.

Interactive CLI:  python -m Backend.main
FastAPI server:   uvicorn Backend.main:app --reload --port 8000
"""
from __future__ import annotations
import asyncio
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Neo4j driver schema warnings for missing nodes/properties
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


# ── FastAPI application ───────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    from .agents.graph import rag_graph
    from .llm.ollama_client import OllamaClient

    app = FastAPI(title="Parrot RAG API", version="0.1.0")

    class QueryRequest(BaseModel):
        query: str
        user_context_username: str | None = None

    class QueryResponse(BaseModel):
        answer: str
        route: str
        num_sources: int
        reasoning: str

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest) -> QueryResponse:
        state = {
            "query": req.query,
            "search_results": [],
        }
        if req.user_context_username:
            state["user_context_username"] = req.user_context_username
        result = await rag_graph.ainvoke(state)
        return QueryResponse(
            answer=result.get("answer", ""),
            route=result.get("route", ""),
            num_sources=len(result.get("search_results", [])),
            reasoning=result.get("reasoning", ""),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "ollama": OllamaClient().is_available(),
        }

except ImportError as e:
    logger.warning("FastAPI not available (%s) API server disabled.", e)
    app = None  # type: ignore


# ── Interactive CLI ────────────────────────────────────────────────────────────
async def interactive_loop(user_context_username: str | None = None) -> None:
    from .agents.graph import rag_graph

    print("\nWelcome to Parrot Agentic RAG. Type your query (Ctrl-C to exit)\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        state = {
            "query": query,
            "search_results": [],
        }
        if user_context_username:
            state["user_context_username"] = user_context_username
        result = await rag_graph.ainvoke(state)
        print(f"\n--- Answer ---")
        print(result.get("answer", "<no answer>"))
        print(
            f"\n[Route: {result.get('route', '?')} | "
            f"Sources: {len(result.get('search_results', []))}]\n"
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parrot Agentic RAG")
    parser.add_argument(
        "--load-user-data",
        action="store_true",
        help="Load user-centric data from activities/feed/messages JSONL files",
    )
    parser.add_argument(
        "--username", help="Username for user-data import/query context")
    parser.add_argument(
        "--activities",
        default="activities.jsonl",
        help="Path to activities JSONL",
    )
    parser.add_argument(
        "--feed",
        default="feed.jsonl",
        help="Path to feed JSONL",
    )
    parser.add_argument(
        "--messages",
        default="messages.jsonl",
        help="Path to messages JSONL",
    )
    return parser


def _run_import_if_requested(args: argparse.Namespace) -> bool:
    if not args.load_user_data:
        return False

    if not args.username:
        raise SystemExit("--username is required when using --load-user-data")

    from pathlib import Path
    from .scripts.fetch_user_data import import_user_data

    stats = asyncio.run(import_user_data(
        username=args.username,
        activities_path=Path(args.activities),
        feed_path=Path(args.feed),
        messages_path=Path(args.messages),
    ))
    print(
        f"Imported for {args.username}: "
        f"{stats.posts} posts, {stats.comments} comments, {stats.messages} messages"
    )
    return True


if __name__ == "__main__":
    cli_args = _build_cli_parser().parse_args()
    if not _run_import_if_requested(cli_args):
        asyncio.run(interactive_loop(user_context_username=cli_args.username))
