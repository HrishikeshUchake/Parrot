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

    import contextlib
    import httpx

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        # On startup: pull all activities from personal_assistant and sync to Neo4j
        personal_assistant_url = os.environ.get(
            "PERSONAL_ASSISTANT_URL", "http://localhost:5002"
        )
        username = os.environ.get("GRAPHRAG_USERNAME", "")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{personal_assistant_url}/all_activities"
                )
                if resp.status_code == 200:
                    activities = resp.json()
                    if activities:
                        logger.info(
                            "Startup sync: %d activities from personal_assistant.",
                            len(activities),
                        )
                        await ingest_activities(
                            IngestRequest(username=username, activities=activities)
                        )
        except Exception as exc:
            logger.warning("Startup sync failed (personal_assistant not ready?): %s", exc)
        yield

    import os
    app = FastAPI(title="Parrot RAG API", version="0.1.0", lifespan=lifespan)

    class QueryRequest(BaseModel):
        query: str
        user_context_username: str | None = None

    class QueryResponse(BaseModel):
        answer: str
        route: str
        num_sources: int
        reasoning: str

    class IngestRequest(BaseModel):
        username: str
        activities: list[dict]

    class IngestResponse(BaseModel):
        status: str
        ingested: int

    @app.post("/ingest_activities", response_model=IngestResponse)
    async def ingest_activities(req: IngestRequest) -> IngestResponse:
        import uuid
        from .services.embedding_service import EmbeddingService
        from .services.vector_store import VectorStore

        svc = EmbeddingService()
        store = VectorStore()

        posts = []
        messages = []

        for act in req.activities:
            source = act.get("source", "")

            if source == "individual_chat":
                text = (act.get("text") or "").replace("TEXT:\n", "").strip()
                if not text:
                    continue
                ts = int(act.get("time") or 0)
                sender = act.get("sender_name", "")
                receiver = act.get("receiver_name", "")
                messages.append({
                    "id": f"msg:{ts}:{sender}:{receiver}",
                    "sender_name": sender,
                    "receiver_name": receiver,
                    "text": text,
                    "date": act.get("date") or "",
                    "time_ms": ts,
                    "source": source,
                })
            else:
                text = (
                    act.get("text_content")
                    or act.get("abs")
                    or act.get("text")
                    or act.get("content")
                    or ""
                ).replace("TEXT:\n", "").strip()
                if not text:
                    continue
                title = act.get("title") or ""
                content = f"{title}\n{text}".strip() if title else text
                raw_tags = act.get("tag_list") or act.get("tags") or []
                if isinstance(raw_tags, str):
                    raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                posts.append({
                    "id": str(act.get("post_id") or act.get("id") or uuid.uuid4()),
                    "content": content,
                    "created_at": act.get("date") or act.get("created_at") or "",
                    "account_id": str(act.get("author_id") or req.username),
                    "account_username": act.get("author_name") or req.username,
                    "account_acct": act.get("author_name") or req.username,
                    "tags": raw_tags,
                    "reblogs_count": 0,
                    "favourites_count": len(act.get("liker_names") or []),
                    "replies_count": len(act.get("comments") or []),
                    "url": act.get("url") or "",
                    "visibility": "private",
                    "language": act.get("language") or "",
                    "source": source or "asmoment",
                    "title": title,
                })

        if posts:
            embeddings = svc.encode_batch([p["content"] for p in posts])
            for post, emb in zip(posts, embeddings):
                post["embedding"] = emb

        if messages:
            embeddings = svc.encode_batch([m["text"] for m in messages])
            for msg, emb in zip(messages, embeddings):
                msg["embedding"] = emb

        if posts or messages:
            store.batch_upsert_user_data(
                posts=posts, messages=messages, comments=[], username=req.username
            )

        return IngestResponse(status="ok", ingested=len(posts) + len(messages))

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
