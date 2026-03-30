"""
fetch_user_data.py
==================
Ingest user-centric social data from local JSONL files and persist to Neo4j.

Supported inputs
----------------
- activities.jsonl (post/activity entries)
- feed.jsonl       (feed post entries)
- messages.jsonl   (direct message entries)

Usage
-----
python -m Backend.scripts.fetch_user_data --username albert336
python -m Backend.scripts.fetch_user_data \
  --username albert336 \
  --activities /path/to/activities.jsonl \
  --feed /path/to/feed.jsonl \
  --messages /path/to/messages.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Backend.config import settings
from Backend.services.embedding_service import EmbeddingService
from Backend.services.vector_store import VectorStore
from Backend.services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)


@dataclass
class ImportStats:
    posts: int = 0
    messages: int = 0
    comments: int = 0
    threads: int = 0
    chunks: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Input file not found: %s", path)
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSON at %s:%d", path, line_no)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _clean_text(value: str) -> str:
    return (value or "").replace("TEXT:\n", "").strip()


def _ms_to_iso(value: Any) -> str:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return ""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.isoformat()


def _parse_tags(tag_list: Any) -> list[str]:
    if isinstance(tag_list, list):
        return [str(t).strip() for t in tag_list if str(t).strip()]
    if isinstance(tag_list, str):
        return [t.strip() for t in tag_list.split(",") if t.strip()]
    return []


def _is_relevant_post(entry: dict[str, Any], username: str) -> bool:
    author = str(entry.get("author_name", ""))
    likers = entry.get("liker_names") or []
    comments = entry.get("comments") or []

    user_commented = any(
        str(c.get("commenter_name", "")) == username for c in comments if isinstance(c, dict)
    )

    return author == username or username in likers or user_commented


def _build_post_record(entry: dict[str, Any]) -> dict[str, Any]:
    text = _clean_text(
        str(entry.get("text_content") or entry.get("abs") or ""))
    title = _clean_text(str(entry.get("title") or ""))
    combined = f"{title}\n{text}".strip()

    return {
        "id": str(entry.get("post_id", "")),
        "content": combined,
        "created_at": _ms_to_iso(entry.get("created_time")),
        "account_id": str(entry.get("author_id", "")),
        "account_username": str(entry.get("author_name", "")),
        "account_acct": str(entry.get("author_name", "")),
        "tags": _parse_tags(entry.get("tag_list")),
        "reblogs_count": 0,
        "favourites_count": len(entry.get("liker_names") or []),
        "replies_count": len(entry.get("comments") or []),
        "url": "",
        "visibility": "private",
        "language": "",
        "source": str(entry.get("source", "")),
        "title": title,
    }


def _build_comment_records(entry: dict[str, Any], username: str) -> list[dict[str, Any]]:
    comments = entry.get("comments") or []
    post_id = str(entry.get("post_id", ""))
    source = str(entry.get("source", ""))

    records: list[dict[str, Any]] = []
    for idx, comment in enumerate(comments):
        if not isinstance(comment, dict):
            continue
        content = _clean_text(str(comment.get("content", "")))
        if not content:
            continue

        commenter_name = str(comment.get("commenter_name", ""))
        # Keep all comments on relevant posts, but include user-related id context.
        comment_id = f"{post_id}:{idx}:{commenter_name}:{comment.get('time', '')}"
        records.append(
            {
                "id": comment_id,
                "post_id": post_id,
                "commenter_name": commenter_name,
                "content": content,
                "time": str(comment.get("time", "")),
                "source": source,
                "is_user_comment": commenter_name == username,
            }
        )
    return records


def _build_message_record(entry: dict[str, Any], username: str) -> dict[str, Any] | None:
    sender = str(entry.get("sender_name", ""))
    receiver = str(entry.get("receiver_name", ""))
    if username not in (sender, receiver):
        return None

    text = _clean_text(str(entry.get("text", "")))
    if not text:
        return None

    ts = int(entry.get("time", 0) or 0)
    msg_id = f"msg:{ts}:{sender}:{receiver}"
    return {
        "id": msg_id,
        "sender_name": sender,
        "receiver_name": receiver,
        "text": text,
        "date": str(entry.get("date", "")),
        "time_ms": ts,
        "source": str(entry.get("source", "individual_chat")),
    }


def _dedupe_by_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in records:
        deduped[str(item.get("id", ""))] = item
    return list(deduped.values())


def _embed_field(records: list[dict[str, Any]], field: str, svc: EmbeddingService) -> None:
    if not records:
        return
    texts = [str(r.get(field, "")) for r in records]
    vectors = svc.encode_batch(texts)
    for row, emb in zip(records, vectors):
        row["embedding"] = emb


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        size = 64
    return [items[i:i + size] for i in range(0, len(items), size)]


async def import_user_data(
    username: str,
    activities_path: Path,
    feed_path: Path,
    messages_path: Path,
) -> ImportStats:
    activities = _read_jsonl(activities_path)
    feed = _read_jsonl(feed_path)
    messages = _read_jsonl(messages_path)

    all_posts_src = activities + feed
    post_records: list[dict[str, Any]] = []
    comment_records: list[dict[str, Any]] = []
    message_records: list[dict[str, Any]] = []

    for row in all_posts_src:
        if not _is_relevant_post(row, username):
            continue
        post_records.append(_build_post_record(row))
        comment_records.extend(_build_comment_records(row, username))

    for row in messages:
        msg = _build_message_record(row, username)
        if msg is not None:
            message_records.append(msg)

    post_records = _dedupe_by_id(post_records)
    comment_records = _dedupe_by_id(comment_records)
    message_records = _dedupe_by_id(message_records)

    svc = EmbeddingService()
    store = VectorStore()
    chunker = ChunkingService()
    batch_size = settings.user_data_import_batch_size

    # 1. Original upserts (backward compatibility)
    for post_chunk in _chunks(post_records, batch_size):
        _embed_field(post_chunk, "content", svc)
        store.batch_upsert_user_data(
            posts=post_chunk,
            messages=[],
            comments=[],
            username=username,
        )

    for comment_chunk in _chunks(comment_records, batch_size):
        _embed_field(comment_chunk, "content", svc)
        store.batch_upsert_user_data(
            posts=[],
            messages=[],
            comments=comment_chunk,
            username=username,
        )

    for message_chunk in _chunks(message_records, batch_size):
        _embed_field(message_chunk, "text", svc)
        store.batch_upsert_user_data(
            posts=[],
            messages=message_chunk,
            comments=[],
            username=username,
        )

    # 2. Derive Conversation Threads
    comments_by_post = defaultdict(list)
    for c in comment_records:
        comments_by_post[c["post_id"]].append(c)

    all_threads = []
    all_chunks = []

    concurrency = max(1, batch_size)

    post_coroutines = [
        chunker.process_post_with_comments(post, comments_by_post[post["id"]])
        for post in post_records
    ]
    for i in range(0, len(post_coroutines), concurrency):
        batch_results = await asyncio.gather(
            *post_coroutines[i:i + concurrency]
        )
        for thread, chunks in batch_results:
            all_threads.append(thread)
            all_chunks.extend(chunks)

    # Group messages by DM conversation
    dm_conversations = defaultdict(list)
    for m in message_records:
        pair = tuple(sorted([m["sender_name"], m["receiver_name"]]))
        dm_conversations[pair].append(m)

    dm_coroutines = [
        chunker.process_dm_thread(f"{pair[0]}_{pair[1]}", msgs)
        for pair, msgs in dm_conversations.items()
    ]
    for i in range(0, len(dm_coroutines), concurrency):
        batch_results = await asyncio.gather(
            *dm_coroutines[i:i + concurrency]
        )
        for thread, chunks in batch_results:
            all_threads.append(thread)
            all_chunks.extend(chunks)

    # 3. Store thread structure
    for thread in all_threads:
        store.upsert_thread(thread.model_dump(), username)

    # 4. Embed and store thread chunks
    chunk_dicts = [c.model_dump() for c in all_chunks]
    for c_chunk in _chunks(chunk_dicts, batch_size):
        _embed_field(c_chunk, "content", svc)
        store.upsert_thread_chunks(c_chunk, username)

    stats = ImportStats(
        posts=len(post_records),
        messages=len(message_records),
        comments=len(comment_records),
        threads=len(all_threads),
        chunks=len(all_chunks),
    )
    logger.info(
        "Imported user data for '%s' -> posts=%d messages=%d comments=%d threads=%d chunks=%d",
        username,
        stats.posts,
        stats.messages,
        stats.comments,
        stats.threads,
        stats.chunks,
    )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import user-centric social data into Neo4j")
    parser.add_argument("--username", required=True,
                        help="Target username context for the data")
    parser.add_argument(
        "--activities",
        default=str(Path.cwd() / "activities.jsonl"),
        help="Path to activities JSONL file",
    )
    parser.add_argument(
        "--feed",
        default=str(Path.cwd() / "feed.jsonl"),
        help="Path to feed JSONL file",
    )
    parser.add_argument(
        "--messages",
        default=str(Path.cwd() / "messages.jsonl"),
        help="Path to messages JSONL file",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
