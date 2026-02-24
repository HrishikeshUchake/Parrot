"""Repository pattern over SQLite – data access layer."""
from __future__ import annotations
import json
import sqlite3
import hashlib
from contextlib import contextmanager
from typing import Generator

from ..config import settings
from .models import Post, QueryCache


@contextmanager
def get_connection(path: str = settings.db_path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class PostRepository:
    """CRUD operations for social media posts."""

    def __init__(self, db_path: str = settings.db_path) -> None:
        self._db_path = db_path

    def get_all(self) -> list[Post]:
        with get_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, title, body, tags, reactions, views, userId FROM posts"
            ).fetchall()
        return [self._row_to_post(r) for r in rows]

    def get_by_id(self, post_id: int) -> Post | None:
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, title, body, tags, reactions, views, userId FROM posts WHERE id = ?",
                (post_id,),
            ).fetchone()
        return self._row_to_post(row) if row else None

    def get_by_ids(self, post_ids: list[int]) -> list[Post]:
        if not post_ids:
            return []
        placeholders = ",".join("?" * len(post_ids))
        with get_connection(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT id, title, body, tags, reactions, views, userId FROM posts WHERE id IN ({placeholders})",
                post_ids,
            ).fetchall()
        return [self._row_to_post(r) for r in rows]

    def keyword_search(self, query: str, limit: int = 10) -> list[Post]:
        """Full-text keyword search using LIKE."""
        pattern = f"%{query}%"
        with get_connection(self._db_path) as conn:
            rows = conn.execute(
                """SELECT id, title, body, tags, reactions, views, userId FROM posts
                   WHERE title LIKE ? OR body LIKE ? OR tags LIKE ?
                   LIMIT ?""",
                (pattern, pattern, pattern, limit),
            ).fetchall()
        return [self._row_to_post(r) for r in rows]

    @staticmethod
    def _row_to_post(row: sqlite3.Row) -> Post:
        tags = json.loads(row["tags"]) if row["tags"] else []
        reactions = json.loads(row["reactions"]) if row["reactions"] else {}
        return Post(
            id=row["id"],
            title=row["title"] or "",
            body=row["body"] or "",
            tags=tags,
            reactions=reactions,
            views=row["views"] or 0,
            user_id=row["userId"] or 0,
        )


class QueryCacheRepository:
    """Persist and look up cached query results."""

    def __init__(self, db_path: str = settings.db_path) -> None:
        self._db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        with get_connection(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> QueryCache | None:
        h = self._hash(query)
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                """SELECT query_hash, results_json, created_at FROM query_cache
                   WHERE query_hash = ?
                     AND created_at > datetime('now', ?)""",
                (h, f"-{settings.cache_ttl_seconds} seconds"),
            ).fetchone()
        if row:
            return QueryCache(
                query_hash=row["query_hash"],
                results_json=row["results_json"],
                created_at=row["created_at"],
            )
        return None

    def set(self, query: str, results_json: str) -> None:
        h = self._hash(query)
        with get_connection(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache (query_hash, results_json) VALUES (?, ?)",
                (h, results_json),
            )
