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

_CREATE_MESSAGE_VECTOR_INDEX = """
    CREATE VECTOR INDEX {index} IF NOT EXISTS
    FOR (m:Message) ON (m.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}
    }}
"""

_CREATE_COMMENT_VECTOR_INDEX = """
    CREATE VECTOR INDEX {index} IF NOT EXISTS
    FOR (c:Comment) ON (c.embedding)
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

_UPSERT_USER_POST = """
    MERGE (p:Post {id: $id})
    SET p.content                 = $content,
        p.created_at              = $created_at,
        p.account_id              = $account_id,
        p.account_username        = $account_username,
        p.account_acct            = $account_acct,
        p.tags_json               = $tags_json,
        p.reblogs_count           = $reblogs_count,
        p.favourites_count        = $favourites_count,
        p.replies_count           = $replies_count,
        p.url                     = $url,
        p.visibility              = $visibility,
        p.language                = $language,
        p.embedding               = $embedding,
        p.source                  = $source,
        p.title                   = $title,
        p.user_context_username   = $user_context_username
    WITH p
    FOREACH (tag IN $tags |
        MERGE (t:Tag {name: tag})
        MERGE (p)-[:HAS_TAG]->(t)
    )
    WITH p
    MERGE (a:Account {id: $account_id})
    ON CREATE SET a.username = $account_username,
                  a.acct     = $account_acct
    MERGE (a)-[:AUTHORED]->(p)
    WITH p
    MERGE (u:User {username: $user_context_username})
    MERGE (u)-[:CAN_SEE]->(p)
"""

_UPSERT_MESSAGE = """
    MERGE (m:Message {id: $id})
    SET m.sender_name            = $sender_name,
        m.receiver_name          = $receiver_name,
        m.text                   = $text,
        m.date                   = $date,
        m.time_ms                = $time_ms,
        m.source                 = $source,
        m.user_context_username  = $user_context_username,
        m.embedding              = $embedding
    WITH m
    MERGE (s:User {username: $sender_name})
    MERGE (r:User {username: $receiver_name})
    MERGE (s)-[:SENT]->(m)
    MERGE (m)-[:TO]->(r)
    WITH m
    MERGE (u:User {username: $user_context_username})
    MERGE (u)-[:CAN_SEE]->(m)
"""

_UPSERT_COMMENT = """
    MERGE (c:Comment {id: $id})
    SET c.post_id                = $post_id,
        c.commenter_name         = $commenter_name,
        c.content                = $content,
        c.time                   = $time,
        c.source                 = $source,
        c.user_context_username  = $user_context_username,
        c.embedding              = $embedding
    WITH c
    MATCH (p:Post {id: $post_id})
    MERGE (p)-[:HAS_COMMENT]->(c)
    WITH c, p
    MERGE (u:User {username: $commenter_name})
    MERGE (u)-[:COMMENTED]->(c)
    WITH c
    MERGE (ctx:User {username: $user_context_username})
    MERGE (ctx)-[:CAN_SEE]->(c)
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

_COUNT_POSTS_BY_USER = """
    MATCH (p:Post)
    WHERE p.user_context_username = $username
    RETURN count(p) AS n
"""

_MESSAGE_VECTOR_SEARCH = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS m, score
    WHERE m.user_context_username = $username
    RETURN m.id                  AS id,
           m.text                AS text,
           m.sender_name         AS sender_name,
           m.receiver_name       AS receiver_name,
           m.date                AS date,
           m.time_ms             AS time_ms,
           m.source              AS source,
           score
"""

_COMMENT_VECTOR_SEARCH = """
    CALL db.index.vector.queryNodes($index, $top_k, $embedding)
    YIELD node AS c, score
    WHERE c.user_context_username = $username
    RETURN c.id                 AS id,
           c.post_id            AS post_id,
           c.content            AS content,
           c.commenter_name     AS commenter_name,
           c.time               AS time,
           c.source             AS source,
           score
"""


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
        self._message_index = settings.neo4j_message_vector_index
        self._comment_index = settings.neo4j_comment_vector_index
        self._ensure_index()

    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Create post/message/comment vector indexes if they don't exist yet."""
        cypher = _CREATE_VECTOR_INDEX.format(
            index=self._index,
            dim=settings.neo4j_embedding_dim,
        )
        message_cypher = _CREATE_MESSAGE_VECTOR_INDEX.format(
            index=self._message_index,
            dim=settings.neo4j_embedding_dim,
        )
        comment_cypher = _CREATE_COMMENT_VECTOR_INDEX.format(
            index=self._comment_index,
            dim=settings.neo4j_embedding_dim,
        )
        with self._driver.session(database=self._db) as session:
            session.run(cypher)
            session.run(message_cypher)
            session.run(comment_cypher)
        logger.info("Neo4j vector index '%s' ready.", self._index)

    # ------------------------------------------------------------------
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = settings.default_top_k,
        where: dict[str, Any] | None = None,
        date_range: tuple[str, str] | None = None,
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

        clauses = []
        if where:
            # Map external filter keys to Neo4j property names
            key_map = {
                "account_id": "account_id",
                "account_username": "account_username",
                "username": "account_username",
                "account_acct": "account_acct",
                "visibility": "visibility",
                "language": "language",
                "user_context_username": "user_context_username",
            }
            for key, val in where.items():
                # Only allow whitelisted keys to prevent invalid property access
                if key not in key_map:
                    continue
                neo4j_key = key_map[key]
                param_key = f"filter_{neo4j_key}"
                clauses.append(f"p.{neo4j_key} = ${param_key}")
                params[param_key] = val

        if date_range:
            start_date, end_date = date_range
            clauses.append(
                "p.created_at >= $start_date AND p.created_at <= $end_date")
            params["start_date"] = start_date
            params["end_date"] = end_date

        if clauses:
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
                        "user_context_username": r.get("user_context_username", ""),
                    },
                    "score": score,
                }
            )
        return results

    # ------------------------------------------------------------------
    def similarity_search_messages(
        self,
        query_embedding: list[float],
        username: str,
        top_k: int = settings.default_top_k,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "index": self._message_index,
            "top_k": top_k,
            "embedding": query_embedding,
            "username": username,
        }
        with self._driver.session(database=self._db) as session:
            records = session.run(_MESSAGE_VECTOR_SEARCH, **params).data()

        return [
            {
                "id": str(r["id"]),
                "document": r.get("text", ""),
                "metadata": {
                    "sender_name": r.get("sender_name", ""),
                    "receiver_name": r.get("receiver_name", ""),
                    "date": r.get("date", ""),
                    "time_ms": int(r.get("time_ms") or 0),
                    "source": r.get("source", ""),
                    "type": "message",
                },
                "score": float(r["score"]),
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    def similarity_search_comments(
        self,
        query_embedding: list[float],
        username: str,
        top_k: int = settings.default_top_k,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "index": self._comment_index,
            "top_k": top_k,
            "embedding": query_embedding,
            "username": username,
        }
        with self._driver.session(database=self._db) as session:
            records = session.run(_COMMENT_VECTOR_SEARCH, **params).data()

        return [
            {
                "id": str(r["id"]),
                "document": r.get("content", ""),
                "metadata": {
                    "post_id": str(r.get("post_id", "")),
                    "commenter_name": r.get("commenter_name", ""),
                    "time": r.get("time", ""),
                    "source": r.get("source", ""),
                    "type": "comment",
                },
                "score": float(r["score"]),
            }
            for r in records
        ]

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
    def batch_upsert_user_data(
        self,
        posts: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        username: str,
    ) -> None:
        """
        Batch upsert user-scoped posts/messages/comments.
        Each input record must already contain an `embedding` field.
        """
        with self._driver.session(database=self._db) as session:
            for post in posts:
                tags = post.get("tags", [])
                if isinstance(tags, str):
                    tags = json.loads(tags)
                session.run(
                    _UPSERT_USER_POST,
                    id=str(post.get("id", "")),
                    content=post.get("content", ""),
                    created_at=post.get("created_at", ""),
                    account_id=str(post.get("account_id", "")),
                    account_username=post.get("account_username", ""),
                    account_acct=post.get("account_acct", ""),
                    tags_json=json.dumps(tags),
                    tags=tags,
                    reblogs_count=int(post.get("reblogs_count", 0)),
                    favourites_count=int(post.get("favourites_count", 0)),
                    replies_count=int(post.get("replies_count", 0)),
                    url=post.get("url", ""),
                    visibility=post.get("visibility", "public"),
                    language=post.get("language", ""),
                    source=post.get("source", ""),
                    title=post.get("title", ""),
                    user_context_username=username,
                    embedding=post["embedding"],
                )

            for message in messages:
                session.run(
                    _UPSERT_MESSAGE,
                    id=str(message.get("id", "")),
                    sender_name=message.get("sender_name", ""),
                    receiver_name=message.get("receiver_name", ""),
                    text=message.get("text", ""),
                    date=message.get("date", ""),
                    time_ms=int(message.get("time_ms", 0)),
                    source=message.get("source", "individual_chat"),
                    user_context_username=username,
                    embedding=message["embedding"],
                )

            for comment in comments:
                session.run(
                    _UPSERT_COMMENT,
                    id=str(comment.get("id", "")),
                    post_id=str(comment.get("post_id", "")),
                    commenter_name=comment.get("commenter_name", ""),
                    content=comment.get("content", ""),
                    time=comment.get("time", ""),
                    source=comment.get("source", ""),
                    user_context_username=username,
                    embedding=comment["embedding"],
                )

        logger.info(
            "Upserted %d posts, %d messages, %d comments for user '%s'.",
            len(posts), len(messages), len(comments), username,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    @property
    def count(self) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run(_COUNT_POSTS).single()
            return result["n"] if result else 0

    def count_for_user(self, username: str) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run(_COUNT_POSTS_BY_USER,
                                 username=username).single()
            return result["n"] if result else 0
