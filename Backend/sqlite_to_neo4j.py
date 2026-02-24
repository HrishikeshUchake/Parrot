"""
sqlite_to_neo4j.py
==================
Ingestion pipeline: SQLite posts → Neo4j graph + vector index.

Run from the repo root:
    python -m Backend.sqlite_to_neo4j

Pre-requisites:
  1. Neo4j 5.11+ running locally (or Aura).  Default bolt://localhost:7687
  2. OLLAMA not required for this step.
  3. A .env file (or environment variables) with NEO4J_PASSWORD if changed.

What it creates in Neo4j
------------------------
  (:Post  {id, title, body, tags_json, views, user_id, embedding})
  (:Tag   {name})
  (:User  {id})
  (:Post)-[:HAS_TAG]->(:Tag)
  (:User)-[:AUTHORED]->(:Post)
  VECTOR INDEX post_embeddings  ON Post(embedding)  cosine / 384 dims
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Allow running as  python -m Backend.sqlite_to_neo4j  OR  python Backend/sqlite_to_neo4j.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from Backend.config import settings


# ── 1. Load posts from SQLite ─────────────────────────────────────────────────
def load_posts_from_sqlite() -> list[dict]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, body, tags, reactions, views, userId FROM posts"
    ).fetchall()
    conn.close()

    posts = []
    for r in rows:
        title = r["title"] or ""
        body = r["body"] or ""
        text = f"{title}\n\n{body}".strip()
        if not text:
            continue
        tags = json.loads(r["tags"]) if r["tags"] else []
        posts.append(
            {
                "id": r["id"],
                "title": title,
                "body": body,
                "tags": tags,
                "tags_json": json.dumps(tags),
                "views": r["views"] or 0,
                "user_id": r["userId"] or 0,
                "text": text,
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
    SET p.title     = $title,
        p.body      = $body,
        p.tags_json = $tags_json,
        p.views     = $views,
        p.user_id   = $user_id,
        p.embedding = $embedding
    WITH p
    FOREACH (tag IN $tags |
        MERGE (t:Tag {name: tag})
        MERGE (p)-[:HAS_TAG]->(t)
    )
    WITH p
    MERGE (u:User {id: $user_id})
    MERGE (u)-[:AUTHORED]->(p)
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
                title=p["title"],
                body=p["body"],
                tags_json=p["tags_json"],
                tags=p["tags"],
                views=p["views"],
                user_id=p["user_id"],
                embedding=p["embedding"],
            )
            if i % 10 == 0 or i == len(posts):
                print(f"  Upserted {i}/{len(posts)} posts...", end="\r")

    print(f"\nDone. {len(posts)} Post nodes in Neo4j.")
    driver.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Connecting to Neo4j at: {settings.neo4j_uri}")
    print(f"Reading SQLite from: {settings.db_path}")

    posts = load_posts_from_sqlite()
    print(f"Loaded {len(posts)} posts from SQLite.")

    posts = embed(posts)
    upsert_to_neo4j(posts)
