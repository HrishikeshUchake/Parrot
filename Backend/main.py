"""
Parrot Agentic RAG System for Social Media Analysis.

Interactive CLI:  python -m Backend.main
FastAPI server:   uvicorn Backend.main:app --reload --port 8000
"""
from __future__ import annotations
import asyncio
import argparse
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Neo4j driver schema warnings for missing nodes/properties
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


# ── FastAPI application ───────────────────────────────────────────────────────────
try:
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel
    from .agents.graph import rag_graph
    from .llm.ollama_client import OllamaClient

    import contextlib
    import httpx

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        from .config import settings
        from collections import Counter
        personal_assistant_url = settings.personal_assistant_url
        username = settings.graphrag_username
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{personal_assistant_url}/all_activities")
                if resp.status_code == 200:
                    activities = resp.json()
                    if activities:
                        # Infer username from activities if not already set
                        if not username:
                            candidate_names: list[str] = []
                            for act in activities:
                                if not isinstance(act, dict):
                                    continue
                                if str(act.get("source", "")) == "individual_chat":
                                    for key in ("sender_name", "receiver_name"):
                                        val = str(act.get(key, "") or "").strip()
                                        if val:
                                            candidate_names.append(val)
                                else:
                                    val = str(act.get("author_name") or act.get("account_username") or "").strip()
                                    if val:
                                        candidate_names.append(val)
                            if candidate_names:
                                username = Counter(candidate_names).most_common(1)[0][0]
                                os.environ["GRAPHRAG_USERNAME"] = username
                                logger.info("Startup sync: inferred username '%s'.", username)

                        logger.info("Startup sync: %d activities from personal_assistant.", len(activities))
                        await ingest_activities(
                            IngestRequest(username=username, activities=activities)
                        )
        except Exception as exc:
            logger.warning("Startup sync failed (personal_assistant not ready?): %s", exc)

        # Fallback: if username still unknown, infer from Neo4j (data already imported)
        if not os.environ.get("GRAPHRAG_USERNAME"):
            try:
                from .services.vector_store import VectorStore
                rows = VectorStore().run_query("""
                    MATCH (u:User)
                    OPTIONAL MATCH (u)-[:HAS_POST]->(p:Post)
                    OPTIONAL MATCH (u)-[:HAS_MESSAGE]->(m:Message)
                    WITH u.username AS username, count(DISTINCT p) + count(DISTINCT m) AS total
                    WHERE total > 0 AND username IS NOT NULL AND username <> ""
                    RETURN username ORDER BY total DESC LIMIT 1
                """)
                if rows:
                    inferred = rows[0]["username"]
                    os.environ["GRAPHRAG_USERNAME"] = inferred
                    logger.info("Startup: inferred username '%s' from Neo4j.", inferred)
            except Exception as exc:
                logger.warning("Startup: Neo4j username inference failed: %s", exc)
        yield

    import os
    app = FastAPI(title="Parrot RAG API", version="0.1.0", lifespan=lifespan)

    class QueryRequest(BaseModel):
        query: str
        user_context_username: str | None = None
        llm_mode: str | None = None

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
        comments = []

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
                    "user_context_username": req.username,
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
                    raw_tags = [t.strip()
                                for t in raw_tags.split(",") if t.strip()]
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
                    "user_context_username": req.username,
                })

                for c in (act.get("comments") or []):
                    if not isinstance(c, dict):
                        continue
                    c_text = c.get("content") or ""
                    if c_text:
                        comments.append({
                            "id": str(c.get("id") or uuid.uuid4()),
                            "post_id": str(act.get("post_id") or act.get("id") or ""),
                            "commenter_name": c.get("commenter_name") or "",
                            "content": c_text,
                            "time": c.get("time") or "",
                            "source": "comment",
                            "user_context_username": req.username,
                        })

        if posts:
            embeddings = svc.encode_batch([p["content"] for p in posts])
            for post, emb in zip(posts, embeddings):
                post["embedding"] = emb

        if messages:
            embeddings = svc.encode_batch([m["text"] for m in messages])
            for msg, emb in zip(messages, embeddings):
                msg["embedding"] = emb

        if comments:
            embeddings = svc.encode_batch([c["content"] for c in comments])
            for comment, emb in zip(comments, embeddings):
                comment["embedding"] = emb

        if posts or messages or comments:
            store.batch_upsert_user_data(
                posts=posts, messages=messages, comments=comments, username=req.username
            )

        return IngestResponse(status="ok", ingested=len(posts) + len(messages) + len(comments))

    @app.post("/deposit_social_activities", response_model=IngestResponse)
    async def deposit_social_activities(payload: Any = Body(None)) -> IngestResponse:
        from .config import settings
        # Backward-compatible endpoint for clients posting to /deposit_social_activities.
        username = os.environ.get("GRAPHRAG_USERNAME", "").strip()

        activities: list[dict] = []

        if isinstance(payload, list):
            activities = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            username = (
                payload.get("username")
                or payload.get("user_context_username")
                or username
            )

            candidate = payload.get("activities")
            if isinstance(candidate, list):
                activities = [x for x in candidate if isinstance(x, dict)]
            elif isinstance(payload.get("chats"), list):
                for item in payload["chats"]:
                    if not isinstance(item, dict):
                        continue
                    normalized = dict(item)
                    normalized.setdefault("source", "individual_chat")
                    if not normalized.get("text"):
                        normalized["text"] = (
                            normalized.get("message")
                            or normalized.get("content")
                            or normalized.get("body")
                            or ""
                        )
                    activities.append(normalized)
            else:
                activity_like = {
                    k: v for k, v in payload.items() if isinstance(v, (str, int, float, list, dict))
                }
                if activity_like:
                    activities = [activity_like]

        if not isinstance(activities, list) or not activities:
            raise HTTPException(
                status_code=400,
                detail="Expected a payload with activities or chats.",
            )

        if not username:
            # Infer the primary user from the payload when the client does not send one.
            candidate_names: list[str] = []

            for act in activities:
                if not isinstance(act, dict):
                    continue

                source = str(act.get("source", "") or "")
                if source == "individual_chat":
                    sender = str(act.get("sender_name", "") or "").strip()
                    receiver = str(act.get("receiver_name", "") or "").strip()
                    if sender:
                        candidate_names.append(sender)
                    if receiver:
                        candidate_names.append(receiver)
                else:
                    author = str(
                        act.get("author_name")
                        or act.get("account_username")
                        or act.get("account_acct")
                        or ""
                    ).strip()
                    if author:
                        candidate_names.append(author)

            if candidate_names:
                from collections import Counter
                username = Counter(candidate_names).most_common(1)[0][0]
            else:
                username = "me"

            # Save the resolved username to .env so the CLI can pick it up automatically
            env_path = Path(__file__).parent / ".env"
            import re
            if env_path.exists():
                content = env_path.read_text("utf-8")
                if "GRAPHRAG_USERNAME=" in content:
                    content = re.sub(
                        r"^GRAPHRAG_USERNAME=.*$", f"GRAPHRAG_USERNAME={username}", content, flags=re.MULTILINE)
                else:
                    if not content.endswith("\n"):
                        content += "\n"
                    content += f"GRAPHRAG_USERNAME={username}\n"
                env_path.write_text(content, "utf-8")
            else:
                env_path.write_text(f"GRAPHRAG_USERNAME={username}\n", "utf-8")

            os.environ["GRAPHRAG_USERNAME"] = username

        return await ingest_activities(
            IngestRequest(username=username, activities=activities)
        )

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest) -> QueryResponse:
        from .config import settings
        user = req.user_context_username or settings.graphrag_username or os.environ.get("GRAPHRAG_USERNAME") or None
        state = {
            "query": req.query,
            "search_results": [],
          }
        if user:
            state["user_context_username"] = user
        if req.llm_mode:
            state["llm_mode"] = req.llm_mode.strip().lower()

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
def _print_pipeline_debug(result: dict) -> None:
    """Print a concise trace of analyzer, router, and retrieval outputs."""
    print("\n--- Pipeline Debug ---")
    print(
        "Analyzer: "
        f"intent={result.get('intent', '?')} | "
        f"complexity={result.get('complexity', '?')} | "
        f"analytics_kind={result.get('analytics_kind', 'none')}"
    )

    entities = result.get("entities") or []
    if entities:
        print(f"Entities: {entities}")

    filters = result.get("filters") or {}
    if filters:
        print(f"Filters: {filters}")

    sub_queries = result.get("sub_queries") or []
    if sub_queries:
        print(f"Sub-queries: {sub_queries}")

    print(f"Router: route={result.get('route', '?')}")
    if result.get("llm_mode"):
        print(f"LLM mode: {result.get('llm_mode')}")

    analytics_payload = result.get("analytics_payload")
    if analytics_payload:
        print(
            "Analytics payload: "
            f"kind={analytics_payload.get('kind', '?')} | "
            f"query_type={analytics_payload.get('query_type', '?')}"
        )
        print(f"Analytics summary: {analytics_payload.get('summary', '')}")

    results = result.get("search_results") or []
    print(f"Retrieved items: {len(results)}")
    for i, item in enumerate(results[:5], 1):
        kind = getattr(item, "result_type", "unknown")
        score = getattr(item, "score", 0.0)
        if kind == "post" and getattr(item, "post", None) is not None:
            post = item.post
            author = post.account_acct or post.account_username or "unknown"
            preview = (post.content or "").replace("\n", " ")[:300]
            print(f"  {i}. post | score={score:.3f} | @{author} | {preview}")
        else:
            content = (getattr(item, "content", "") or "").replace("\n", " ")[:300]
            print(f"  {i}. {kind} | score={score:.3f} | {content}")

    reasoning = result.get("reasoning")
    if reasoning:
        print(f"Reasoning: {reasoning}")

    privacy_debug = result.get("privacy_debug") or {}
    if privacy_debug:
        print(
            "Privacy: "
            f"enabled={privacy_debug.get('privacy_enabled')} | "
            f"anonymizer={privacy_debug.get('privacy_anonymizer')} | "
            f"engine={privacy_debug.get('privatizer_class')} | "
            f"pii_detected={privacy_debug.get('pii_detected')}"
        )
        mappings = privacy_debug.get("mappings") or {}
        if mappings:
            print("PII mappings:")
            for token, original in mappings.items():
                print(f"  {token} -> {original!r}")

    anonymized_context = privacy_debug.get("anonymized_context")
    remote_prompt = privacy_debug.get("remote_prompt")
    anonymized_answer = privacy_debug.get("anonymized_answer")
    restored_answer = result.get("answer")

    if anonymized_context is not None:
        print("\nAnonymized context sent to remote LLM:")
        print("=" * 70)
        print(anonymized_context)
        print("=" * 70)
    if remote_prompt is not None:
        print("\nFull remote prompt sent to remote LLM:")
        print("=" * 70)
        print(remote_prompt)
        print("=" * 70)
    if anonymized_answer is not None:
        print("\nPrivatized retrieved response (anonymized answer):")
        print("=" * 70)
        print(anonymized_answer)
        print("=" * 70)
    if restored_answer is not None:
        print("\nDeanonymized retrieved answer:")
        print("=" * 70)
        print(restored_answer)
        print("=" * 70)
        
_CONFIRMATIONS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "y"}

def _cli_source_preview(results: list[Any]) -> list[str]:
    lines = []
    for i, r in enumerate(results[:5], start=1):
        result_type = getattr(r, "result_type", "unknown")
        item_id = getattr(r, "item_id", "")
        score = getattr(r, "score", 0.0)
        metadata = getattr(r, "metadata", {}) or {}

        if result_type == "message":
            label = f"{metadata.get('sender_name', '')}->{metadata.get('receiver_name', '')}"
        elif result_type == "thread":
            thread = getattr(r, "thread", None)
            label = f"participants={getattr(thread, 'participants', [])}"
        elif result_type == "comment":
            label = f"commenter={metadata.get('commenter_name', '')}"
        elif result_type == "post":
            post = getattr(r, "post", None)
            label = f"author={getattr(post, 'account_username', '') if post else ''}"
        else:
            label = ""

        lines.append(
            f"  {i}. {result_type} | id={item_id} | score={score:.3f} | {label}"
        )
    return lines

async def interactive_loop(
    user_context_username: str | None = None,
    debug_pipeline: bool = False,
    llm_mode: str | None = None,
) -> None:
    import re as _re
    from .agents.graph import rag_graph

    print("\nWelcome to Parrot Agentic RAG. Type your query (Ctrl-C to exit)\n")

    pending_correction: tuple[str, str] | None = None  # (original_query, corrected_query)

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # If the user confirms a "did you mean" suggestion, rerun with corrected query
        if pending_correction and query.lower() in _CONFIRMATIONS:
            query = pending_correction[1]
            print(f"(Running: {query})")
        pending_correction = None

        state = {
            "query": query,
            "search_results": [],
        }
        if user_context_username:
            state["user_context_username"] = user_context_username
        if llm_mode:
            state["llm_mode"] = llm_mode
        if debug_pipeline:
            state["debug_pipeline"] = True
        result = await rag_graph.ainvoke(state)
        answer = result.get("answer", "<no answer>")

        if debug_pipeline:
            _print_pipeline_debug(result)

        # Detect "did you mean" answer and store corrected query for next turn
        m = _re.search(r"Did you mean one of these: \*\*(\w+)\*\*", answer)
        if m:
            suggestion = m.group(1)
            original_partner = _re.search(r"\bwith\s+@?(\w+)\b", query, _re.IGNORECASE)
            if original_partner:
                corrected = _re.sub(
                    r"@?" + _re.escape(original_partner.group(1)),
                    suggestion,
                    query,
                    count=1,
                    flags=_re.IGNORECASE,
                )
                pending_correction = (query, corrected)
        print(f"\n--- Answer ---")
        print(answer)
        sources = result.get("search_results", [])
        print(f"\n[Route: {result.get('route', '?')} | Sources: {len(sources)}]")
        for line in _cli_source_preview(sources):
            print(line)
        print()


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
    parser.add_argument(
        "--debug-pipeline",
        action="store_true",
        help="Print query analyzer, router, and retrieval details for each query",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["remote", "local"],
        default=None,
        help="Choose synthesis mode explicitly: remote (privacy + remote prompt) or local",
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

    # Save the username to .env for the CLI to pick up automatically
    env_path = Path(__file__).parent / ".env"
    import re
    if env_path.exists():
        content = env_path.read_text("utf-8")
        if "GRAPHRAG_USERNAME=" in content:
            content = re.sub(r"^GRAPHRAG_USERNAME=.*$",
                             f"GRAPHRAG_USERNAME={args.username}", content, flags=re.MULTILINE)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += f"GRAPHRAG_USERNAME={args.username}\n"
        env_path.write_text(content, "utf-8")
    else:
        env_path.write_text(f"GRAPHRAG_USERNAME={args.username}\n", "utf-8")

    print(
        f"Imported for {args.username}: "
        f"{stats.posts} posts, {stats.comments} comments, {stats.messages} messages"
    )
    return True


if __name__ == "__main__":
    cli_args = _build_cli_parser().parse_args()
    if not _run_import_if_requested(cli_args):
        context_user = cli_args.username or os.environ.get("GRAPHRAG_USERNAME") or None
        if not context_user:
            try:
                import httpx as _httpx
                from .config import settings as _settings
                resp = _httpx.get(f"{_settings.personal_assistant_url}/me", timeout=5)
                if resp.status_code == 200:
                    context_user = resp.json().get("username") or None
                    if context_user:
                        logger.info("CLI: got username '%s' from personal_assistant.", context_user)
            except Exception as exc:
                logger.warning("CLI: personal_assistant /me failed: %s", exc)

        asyncio.run(
            interactive_loop(
                user_context_username=context_user,
                debug_pipeline=cli_args.debug_pipeline,
                llm_mode=cli_args.llm_mode,
            )
        )