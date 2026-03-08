"""
sqlite_to_neo4j.py
==================
Ingestion pipeline: SQLite Mastodon posts → Neo4j graph + vector index.

Run from the repo root:
    python -m Backend.scripts.sqlite_to_neo4j

Pre-requisites:
  1. Neo4j 5.11+ running locally (or Aura).  Default bolt://localhost:7687
  2. Mastodon statuses already fetched into SQLite via:
         python -m Backend.scripts.fetch_mastodon
  3. A .env file (or environment variables) with NEO4J_PASSWORD if changed.

What it creates in Neo4j
------------------------
  (:Post    {id, content, created_at, account_id, account_username,
             account_acct, tags_json, reblogs_count, favourites_count,
             replies_count, url, visibility, language, embedding})
  (:Tag     {name})
  (:Account {id, username, display_name, acct})
  (:Post)-[:HAS_TAG]->(:Tag)
  (:Account)-[:AUTHORED]->(:Post)
  VECTOR INDEX post_embeddings  ON Post(embedding)  cosine / 384 dims
"""
from __future__ import annotations
from Backend.config import settings
import json
import sqlite3
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Allow running as  python -m Backend.scripts.sqlite_to_neo4j  OR  python Backend/scripts/sqlite_to_neo4j.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── 1. Load posts from SQLite ─────────────────────────────────────────────────

def load_posts_from_sqlite() -> list[dict]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, content, created_at,
                  account_id, account_username, account_display_name, account_acct,
                  tags, reblogs_count, favourites_count, replies_count,
                  url, visibility, language
           FROM posts"""
    ).fetchall()
    conn.close()

    posts = []
    for r in rows:
        content = (r["content"] or "").strip()
        if not content:
            continue
        tags = json.loads(r["tags"]) if r["tags"] else []
        posts.append(
            {
                "id": str(r["id"]),
                "content": content,
                "created_at": r["created_at"] or "",
                "account_id": str(r["account_id"] or ""),
                "account_username": r["account_username"] or "",
                "account_display_name": r["account_display_name"] or "",
                "account_acct": r["account_acct"] or "",
                "tags": tags,
                "tags_json": json.dumps(tags),
                "reblogs_count": int(r["reblogs_count"] or 0),
                "favourites_count": int(r["favourites_count"] or 0),
                "replies_count": int(r["replies_count"] or 0),
                "url": r["url"] or "",
                "visibility": r["visibility"] or "public",
                "language": r["language"] or "",
                "text": content,
            }
        )
    return posts


# ── 2. Generate embeddings ────────────────────────────────────────────────────
def embed(posts: list[dict]) -> list[dict]:
    print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    texts = [p["text"] for p in posts]
    vecs = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()
    for p, v in zip(posts, vecs):
        p["embedding"] = v
    return posts


# ── 3. Upsert into Neo4j ──────────────────────────────────────────────────────
_ENSURE_VECTOR_INDEX = """
    CREATE VECTOR INDEX {index} IF NOT EXISTS
    FOR (p:Post) ON (p.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}
    }}
"""

_UPSERT_POST = """
    MERGE (p:Post {id: $id})
    SET p.content              = $content,
        p.created_at           = $created_at,
        p.account_id           = $account_id,
        p.account_username     = $account_username,
        p.account_acct         = $account_acct,
        p.tags_json            = $tags_json,
        p.reblogs_count        = $reblogs_count,
        p.favourites_count     = $favourites_count,
        p.replies_count        = $replies_count,
        p.url                  = $url,
        p.visibility           = $visibility,
        p.language             = $language,
        p.embedding            = $embedding
    WITH p
    FOREACH (tag IN $tags |
        MERGE (t:Tag {name: tag})
        MERGE (p)-[:HAS_TAG]->(t)
    )
    WITH p
    MERGE (a:Account {id: $account_id})
    ON CREATE SET a.username     = $account_username,
                  a.display_name = $account_display_name,
                  a.acct         = $account_acct
    MERGE (a)-[:AUTHORED]->(p)
"""


def upsert_to_neo4j(posts: list[dict]) -> None:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    db = settings.neo4j_database

    with driver.session(database=db) as session:
        # Ensure vector index exists
        session.run(
            _ENSURE_VECTOR_INDEX.format(
                index=settings.neo4j_vector_index,
                dim=settings.neo4j_embedding_dim,
            )
        )
        print(f"Vector index '{settings.neo4j_vector_index}' ready.")

        # Upsert posts
        for i, p in enumerate(posts, 1):
            session.run(
                _UPSERT_POST,
                id=p["id"],
                content=p["content"],
                created_at=p["created_at"],
                account_id=p["account_id"],
                account_username=p["account_username"],
                account_display_name=p["account_display_name"],
                account_acct=p["account_acct"],
                tags_json=p["tags_json"],
                tags=p["tags"],
                reblogs_count=p["reblogs_count"],
                favourites_count=p["favourites_count"],
                replies_count=p["replies_count"],
                url=p["url"],
                visibility=p["visibility"],
                language=p["language"],
                embedding=p["embedding"],
            )
            if i % 10 == 0 or i == len(posts):
                print(f"  Upserted {i}/{len(posts)} posts...", end="\r")

    print(f"\nDone. {len(posts)} Post nodes in Neo4j.")
    driver.close()


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to Neo4j at: {settings.neo4j_uri}")
    print(f"Reading SQLite from: {settings.db_path}")

    posts = load_posts_from_sqlite()
    if not posts:
        print("No posts found in SQLite. Run `python -m Backend.fetch_mastodon` first.")
    else:
        print(f"Loaded {len(posts)} statuses from SQLite.")
        posts = embed(posts)
        upsert_to_neo4j(posts)
