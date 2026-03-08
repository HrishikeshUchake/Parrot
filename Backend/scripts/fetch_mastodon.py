"""
fetch_mastodon.py
=================
Ingestion – fetch statuses from a Mastodon instance, embed them, and persist
them directly into Neo4j (no SQLite intermediate step).

Usage
-----
    # From the repo root:
    python -m Backend.scripts.fetch_mastodon

    # Override instance and target count at runtime:
    MASTODON_INSTANCE_URL=https://fosstodon.org \\
    MASTODON_FETCH_LIMIT=500 \\
    python -m Backend.scripts.fetch_mastodon

Configuration (config.py / .env)
---------------------------------
  MASTODON_INSTANCE_URL    e.g. https://mastodon.social  (default)
  MASTODON_ACCESS_TOKEN    personal access token (optional; needed for
                             authenticated timelines / higher rate limits)
  MASTODON_FETCH_LIMIT     total number of statuses to ingest (default 200)
  MASTODON_PAGE_SIZE       results per API page, max 40      (default 40)
  MASTODON_LOCAL_ONLY      set "true" to restrict to the local instance

What gets stored in Neo4j
--------------------------
  (:Post {id, content, created_at, account_id, account_username,
          account_display_name, account_acct, tags_json,
          reblogs_count, favourites_count, replies_count,
          url, visibility, language, embedding})
  (:Tag     {name})
  (:Account {id, username, display_name, acct})
  (:Post)-[:HAS_TAG]->(:Tag)
  (:Account)-[:AUTHORED]->(:Post)
  VECTOR INDEX post_embeddings ON Post(embedding)  cosine / 384 dims
"""
from __future__ import annotations
from Backend.config import settings

import html
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"\s{2,}")


def _strip_html(raw: str) -> str:
    """Remove HTML tags and decode entities, collapse whitespace."""
    without_tags = _TAG_RE.sub(" ", raw or "")
    decoded = html.unescape(without_tags)
    return _MULTI_SPACE.sub(" ", decoded).strip()


def _parse_datetime(value: Any) -> str:
    """Normalise Mastodon datetime to ISO-8601 string (UTC)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value) if value else ""


# ── Mastodon API client (thin wrapper over requests) ─────────────────────────

class MastodonClient:
    """Minimal public/authenticated Mastodon REST client."""

    _HEADERS = {
        "User-Agent": "Parrot-RAG/1.0 (+https://github.com/your-org/parrot)"}

    def __init__(
        self,
        instance_url: str,
        access_token: str = "",
    ) -> None:
        self._base = instance_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(self._HEADERS)
        if access_token:
            self._session.headers["Authorization"] = f"Bearer {access_token}"

    def public_timeline(
        self,
        limit: int = 40,
        local: bool = False,
        max_id: str | None = None,
    ) -> list[dict]:
        """Fetch one page from the public (or local) timeline."""
        params: dict[str, Any] = {"limit": min(limit, 40)}
        if local:
            params["local"] = "true"
        if max_id:
            params["max_id"] = max_id
        resp = self._session.get(
            f"{self._base}/api/v1/timelines/public",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def home_timeline(
        self,
        limit: int = 40,
        max_id: str | None = None,
    ) -> list[dict]:
        """Fetch one page from the authenticated home timeline."""
        params: dict[str, Any] = {"limit": min(limit, 40)}
        if max_id:
            params["max_id"] = max_id
        resp = self._session.get(
            f"{self._base}/api/v1/timelines/home",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def trending_statuses(
        self,
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict]:
        """
        Fetch trending statuses – publicly accessible without auth on most
        instances (including mastodon.social).  Uses offset-based pagination.
        """
        params: dict[str, Any] = {"limit": min(limit, 40), "offset": offset}
        resp = self._session.get(
            f"{self._base}/api/v1/trends/statuses",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ── Status normalisation ──────────────────────────────────────────────────────

def _normalise(status: dict) -> dict | None:
    """
    Convert a raw Mastodon status dict into a flat dict ready for Neo4j.
    Returns None for pure boosts so they are not double-counted.
    """
    # Skip pure boosts – the reblogged post will surface on its own
    if status.get("reblog") and not (status.get("content") or "").strip():
        return None

    content_raw = status.get("content") or ""
    content = _strip_html(content_raw)
    if not content:
        return None

    acct: dict = status.get("account") or {}
    tags: list[str] = [t["name"]
                       for t in (status.get("tags") or []) if t.get("name")]

    return {
        "id": str(status["id"]),
        "content": content,
        "created_at": _parse_datetime(status.get("created_at", "")),
        "account_id": str(acct.get("id", "")),
        "account_username": acct.get("username", ""),
        "account_display_name": acct.get("display_name", "") or acct.get("username", ""),
        "account_acct": acct.get("acct", ""),
        # list[str] – Neo4j stores natively; tags_json added at upsert time
        "tags": tags,
        "reblogs_count": int(status.get("reblogs_count") or 0),
        "favourites_count": int(status.get("favourites_count") or 0),
        "replies_count": int(status.get("replies_count") or 0),
        "url": status.get("url") or "",
        "visibility": status.get("visibility") or "public",
        "language": status.get("language") or "",
    }


# ── Main ingestion loop ───────────────────────────────────────────────────────

def fetch_and_store(
    instance_url: str = settings.mastodon_instance_url,
    access_token: str = settings.mastodon_access_token,
    total: int = settings.mastodon_fetch_limit,
    page_size: int = settings.mastodon_page_size,
    local_only: bool = settings.mastodon_local_only,
    embed_batch_size: int = settings.embedding_batch_size,
) -> int:
    """
    Fetch up to *total* statuses from *instance_url*, embed them, and upsert
    directly into Neo4j.  Returns the number of posts inserted/updated.
    """
    # Lazy import to avoid loading Neo4j/sentence-transformers at module level
    from Backend.services.embedding_service import EmbeddingService
    from Backend.services.vector_store import VectorStore

    emb_svc = EmbeddingService()
    store = VectorStore()
    client = MastodonClient(instance_url, access_token)

    fetched = 0
    inserted = 0
    max_id: str | None = None
    pending: list[dict] = []          # accumulated normalised records
    # timeline_mode: "home" (auth) | "public" | "trending" (fallback)
    timeline_mode = "home" if bool(access_token) else "public"

    print(f"Fetching up to {total} statuses from {instance_url} …")

    def _flush(batch: list[dict]) -> int:
        """Embed a batch of records and upsert into Neo4j. Returns count upserted."""
        if not batch:
            return 0
        texts = [r["content"] for r in batch]
        embeddings = emb_svc.encode_batch(texts)
        ids = [r["id"] for r in batch]
        documents = texts
        metadatas = [
            {
                "content": r["content"],
                "created_at": r["created_at"],
                "account_id": r["account_id"],
                "account_username": r["account_username"],
                "account_display_name": r["account_display_name"],
                "account_acct": r["account_acct"],
                "tags": r["tags"],               # list[str]
                "tags_json": json.dumps(r["tags"]),
                "reblogs_count": r["reblogs_count"],
                "favourites_count": r["favourites_count"],
                "replies_count": r["replies_count"],
                "url": r["url"],
                "visibility": r["visibility"],
                "language": r["language"],
            }
            for r in batch
        ]
        store.upsert(ids=ids, documents=documents,
                     metadatas=metadatas, embeddings=embeddings)
        return len(batch)

    while fetched < total:
        remaining = total - fetched
        batch_size = min(page_size, remaining, 40)

        try:
            if timeline_mode == "home":
                page: list[dict] = client.home_timeline(
                    limit=batch_size, max_id=max_id
                )
            elif timeline_mode == "public":
                page = client.public_timeline(
                    limit=batch_size,
                    local=local_only,
                    max_id=max_id,
                )
            else:  # trending – offset-based
                page = client.trending_statuses(
                    limit=batch_size, offset=fetched
                )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            if status_code == 422 and timeline_mode == "public":
                print(
                    "\n  Public timeline unavailable without auth (422) – "
                    "falling back to /api/v1/trends/statuses …"
                )
                timeline_mode = "trending"
                fetched = 0
                max_id = None
                continue
            print(
                f"  HTTP error: {exc} – stopping pagination.", file=sys.stderr)
            break
        except requests.RequestException as exc:
            print(
                f"  Network error: {exc} – stopping pagination.", file=sys.stderr)
            break

        if not page:
            print("  Empty page returned – reached end of timeline.")
            break

        for raw in page:
            record = _normalise(raw)
            if record is None:
                continue
            pending.append(record)

            # Flush whenever we have a full embed batch
            if len(pending) >= embed_batch_size:
                inserted += _flush(pending)
                pending = []

        fetched += len(page)
        if timeline_mode != "trending":
            max_id = str(page[-1]["id"])

        print(
            f"  Fetched {fetched}/{total}  |  stored {inserted} statuses …", end="\r")
        time.sleep(0.5)   # be polite to the instance

    # Flush any remaining records
    inserted += _flush(pending)

    print(f"\nDone. {inserted} Mastodon statuses upserted into Neo4j.")
    return inserted


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    fetch_and_store()
