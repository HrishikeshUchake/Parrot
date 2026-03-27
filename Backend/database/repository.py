"""Repository pattern over Neo4j data access layer."""
from __future__ import annotations
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase

from ..config import settings
from .models import Post, QueryCache

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(path: str) -> Generator[sqlite3.Connection, None, None]:
    """SQLite connection context manager (used by QueryCacheRepository)."""
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


_RETURN_POST_FIELDS = """
    RETURN p.id                   AS id,
           p.content              AS content,
           p.created_at           AS created_at,
           p.account_id           AS account_id,
           p.account_username     AS account_username,
           p.account_display_name AS account_display_name,
           p.account_acct         AS account_acct,
           p.tags_json            AS tags_json,
           p.reblogs_count        AS reblogs_count,
           p.favourites_count     AS favourites_count,
           p.replies_count        AS replies_count,
           p.url                  AS url,
           p.visibility           AS visibility,
           p.language             AS language
"""

_GET_BY_IDS = f"""
    MATCH (p:Post)
    WHERE p.id IN $ids
    {_RETURN_POST_FIELDS}
"""

# NOTE: `_KEYWORD_SEARCH` requires the Neo4j fulltext index "post_fulltext",
# which is created in `vector_store.py`; if the index is missing, this query
# will fail at runtime.
_KEYWORD_SEARCH = f"""
    CALL db.index.fulltext.queryNodes("post_fulltext", $search_term) YIELD node AS p, score
    {_RETURN_POST_FIELDS}
    LIMIT $limit
"""


class PostRepository:
    """Read operations for Mastodon posts stored in Neo4j."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._db = settings.neo4j_database

    def get_by_ids(self, post_ids: list[str]) -> list[Post]:
        if not post_ids:
            return []
        with self._driver.session(database=self._db) as session:
            records = session.run(
                _GET_BY_IDS, ids=[str(pid) for pid in post_ids]
            ).data()
        return [self._record_to_post(r) for r in records]

    def keyword_search(self, query: str, limit: int = 10) -> list[Post]:
        """Case-insensitive keyword search over content, tags and account fields."""
        with self._driver.session(database=self._db) as session:
            records = session.run(
                _KEYWORD_SEARCH, search_term=query, limit=limit
            ).data()
        return [self._record_to_post(r) for r in records]

    def count(self) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run(
                "MATCH (p:Post) RETURN count(p) AS n").single()
            return result["n"] if result else 0

    @staticmethod
    def _record_to_post(r: dict[str, Any]) -> Post:
        tags = json.loads(r["tags_json"]) if r.get("tags_json") else []
        return Post(
            id=str(r["id"]),
            content=r.get("content") or "",
            created_at=r.get("created_at") or "",
            account_id=str(r.get("account_id") or ""),
            account_username=r.get("account_username") or "",
            account_display_name=r.get("account_display_name") or "",
            account_acct=r.get("account_acct") or "",
            tags=tags,
            reblogs_count=int(r.get("reblogs_count") or 0),
            favourites_count=int(r.get("favourites_count") or 0),
            replies_count=int(r.get("replies_count") or 0),
            url=r.get("url") or "",
            visibility=r.get("visibility") or "public",
            language=r.get("language") or "",
        )


class QueryCacheRepository:
    """Persist and look up cached query results (SQLite sidecar)."""

    def __init__(self, db_path: str = "query_cache.db") -> None:
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
