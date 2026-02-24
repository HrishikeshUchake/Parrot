"""VectorStore - Neo4j graph database with native vector index."""
from __future__ import annotations
import json
import logging
from typing import Any

from neo4j import GraphDatabase, Driver

from ..config import settings

logger = logging.getLogger(__name__)

_CREATE_VECTOR_INDEX = """
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

_VECTOR_SEARCH = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS p, score
    RETURN p.id        AS id,
           p.title     AS title,
           p.body      AS body,
           p.tags_json AS tags_json,
           p.views     AS views,
           p.user_id   AS user_id,
           score
"""

_VECTOR_SEARCH_FILTERED = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS p, score
    WHERE {where_clause}
    RETURN p.id        AS id,
           p.title     AS title,
           p.body      AS body,
           p.tags_json AS tags_json,
           p.views     AS views,
           p.user_id   AS user_id,
           score
"""

_COUNT_POSTS = "MATCH (p:Post) RETURN count(p) AS n"


class VectorStore:
    """
    Neo4j-backed vector store.

    Graph schema
    ============
    (:Post  {id, title, body, tags_json, views, user_id, embedding})
    (:Tag   {name})
    (:User  {id})
    (:Post)-[:HAS_TAG]->(:Tag)
    (:User)-[:AUTHORED]->(:Post)

    Vector index
    ============
    Name  : settings.neo4j_vector_index
    On    : Post.embedding  (cosine similarity)
    Dims  : settings.neo4j_embedding_dim
    """

    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._db = settings.neo4j_database
        self._index = settings.neo4j_vector_index
        self._ensure_index()

    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Create the vector index if it doesn't exist yet."""
        cypher = _CREATE_VECTOR_INDEX.format(
            index=self._index,
            dim=settings.neo4j_embedding_dim,
        )
        with self._driver.session(database=self._db) as session:
            session.run(cypher)
        logger.info("Neo4j vector index '%s' ready.", self._index)

    # ------------------------------------------------------------------
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = settings.default_top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns a list of dicts with keys:
          id, document, metadata, score
        Compatible with the same interface previously used by ChromaDB.
        """
        params: dict[str, Any] = {
            "index": self._index,
            "top_k": top_k,
            "embedding": query_embedding,
        }

        if where:
            clauses = []
            # Map external filter keys to Neo4j property names
            key_map = {"userId": "user_id", "user_id": "user_id"}
            for key, val in where.items():
                neo4j_key = key_map.get(key, key)
                param_key = f"filter_{neo4j_key}"
                clauses.append(f"p.{neo4j_key} = ${param_key}")
                params[param_key] = val
            cypher = _VECTOR_SEARCH_FILTERED.format(
                where_clause=" AND ".join(clauses)
            )
        else:
            cypher = _VECTOR_SEARCH

        with self._driver.session(database=self._db) as session:
            records = session.run(cypher, **params).data()

        results = []
        for r in records:
            score = float(r["score"])
            tags = json.loads(r["tags_json"]) if r["tags_json"] else []
            results.append(
                {
                    "id": int(r["id"]),
                    "document": f"{r['title']}\n\n{r['body']}".strip(),
                    "metadata": {
                        "title": r["title"],
                        "tags": tags,
                        "views": r["views"],
                        "user_id": r["user_id"],
                    },
                    "score": score,
                }
            )
        return results

    # ------------------------------------------------------------------
    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Batch upsert Post nodes with their embeddings and graph relationships."""
        with self._driver.session(database=self._db) as session:
            for post_id, meta, emb in zip(ids, metadatas, embeddings):
                tags = meta.get("tags", [])
                if isinstance(tags, str):
                    # tags may come in as JSON string from old ingestion scripts
                    tags = json.loads(tags)
                session.run(
                    _UPSERT_POST,
                    id=int(post_id),
                    title=meta.get("title", ""),
                    body=meta.get("body", ""),
                    tags_json=json.dumps(tags),
                    tags=tags,
                    views=meta.get("views", 0),
                    user_id=meta.get("userId") or meta.get("user_id", 0),
                    embedding=emb,
                )
        logger.info("Upserted %d Post nodes into Neo4j.", len(ids))

    # ------------------------------------------------------------------
    def graph_neighbors(
        self, post_id: int, hops: int = 1
    ) -> list[dict[str, Any]]:
        """
        Return Posts connected to `post_id` through shared tags or authorship.
        Useful for graph-aware retrieval (e.g. 'find similar posts by same author').
        """
        cypher = """
            MATCH (p:Post {id: $id})
            MATCH (p)-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(neighbor:Post)
            WHERE neighbor.id <> $id
            RETURN DISTINCT neighbor.id   AS id,
                            neighbor.title AS title,
                            count(t)       AS shared_tags
            ORDER BY shared_tags DESC
            LIMIT 10
        """
        with self._driver.session(database=self._db) as session:
            return session.run(cypher, id=post_id).data()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    @property
    def count(self) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run(_COUNT_POSTS).single()
            return result["n"] if result else 0
