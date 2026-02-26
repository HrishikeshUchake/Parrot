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
                  a.acct         = $account_acct
    MERGE (a)-[:AUTHORED]->(p)
"""

_VECTOR_SEARCH = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS p, score
    RETURN p.id               AS id,
           p.content          AS content,
           p.account_id       AS account_id,
           p.account_username AS account_username,
           p.account_acct     AS account_acct,
           p.tags_json        AS tags_json,
           p.reblogs_count    AS reblogs_count,
           p.favourites_count AS favourites_count,
           p.replies_count    AS replies_count,
           p.visibility       AS visibility,
           p.language         AS language,
           score
"""

_VECTOR_SEARCH_FILTERED = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS p, score
    WHERE {where_clause}
    RETURN p.id               AS id,
           p.content          AS content,
           p.account_id       AS account_id,
           p.account_username AS account_username,
           p.account_acct     AS account_acct,
           p.tags_json        AS tags_json,
           p.reblogs_count    AS reblogs_count,
           p.favourites_count AS favourites_count,
           p.replies_count    AS replies_count,
           p.visibility       AS visibility,
           p.language         AS language,
           score
"""

_COUNT_POSTS = "MATCH (p:Post) RETURN count(p) AS n"


class VectorStore:
    """
    Neo4j-backed vector store.

    Graph schema
    ============
    (:Post    {id, content, created_at, account_id, account_username,
               account_acct, tags_json, reblogs_count, favourites_count,
               replies_count, url, visibility, language, embedding})
    (:Tag     {name})
    (:Account {id, username, acct})
    (:Post)-[:HAS_TAG]->(:Tag)
    (:Account)-[:AUTHORED]->(:Post)

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
            key_map = {
                "account_id": "account_id",
                "account_username": "account_username",
                "account_acct": "account_acct",
                "visibility": "visibility",
                "language": "language",
            }
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
                    "id": str(r["id"]),
                    "document": (r["content"] or "").strip(),
                    "metadata": {
                        "account_id": r["account_id"],
                        "account_username": r["account_username"],
                        "account_acct": r["account_acct"],
                        "tags": tags,
                        "reblogs_count": r["reblogs_count"],
                        "favourites_count": r["favourites_count"],
                        "replies_count": r["replies_count"],
                        "visibility": r["visibility"],
                        "language": r["language"],
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
                    tags = json.loads(tags)
                session.run(
                    _UPSERT_POST,
                    id=str(post_id),
                    content=meta.get("content", ""),
                    created_at=meta.get("created_at", ""),
                    account_id=str(meta.get("account_id", "")),
                    account_username=meta.get("account_username", ""),
                    account_acct=meta.get("account_acct", ""),
                    tags_json=json.dumps(tags),
                    tags=tags,
                    reblogs_count=int(meta.get("reblogs_count", 0)),
                    favourites_count=int(meta.get("favourites_count", 0)),
                    replies_count=int(meta.get("replies_count", 0)),
                    url=meta.get("url", ""),
                    visibility=meta.get("visibility", "public"),
                    language=meta.get("language", ""),
                    embedding=emb,
                )
        logger.info("Upserted %d Post nodes into Neo4j.", len(ids))

    # ------------------------------------------------------------------
    def graph_neighbors(
        self, post_id: str, hops: int = 1
    ) -> list[dict[str, Any]]:
        """
        Return Posts connected to `post_id` through shared tags or authorship.
        Useful for graph-aware retrieval (e.g. 'find similar posts by same author').
        """
        cypher = """
            MATCH (p:Post {id: $id})
            MATCH (p)-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(neighbor:Post)
            WHERE neighbor.id <> $id
            RETURN DISTINCT neighbor.id            AS id,
                            neighbor.account_acct  AS account_acct,
                            count(t)               AS shared_tags
            ORDER BY shared_tags DESC
            LIMIT 10
        """
        with self._driver.session(database=self._db) as session:
            return session.run(cypher, id=str(post_id)).data()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    @property
    def count(self) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run(_COUNT_POSTS).single()
            return result["n"] if result else 0
