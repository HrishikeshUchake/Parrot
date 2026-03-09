"""
Recompute embeddings for existing Post/Message/Comment nodes in Neo4j.

Usage:
    python -m Backend.scripts.migrate_embeddings
    python -m Backend.scripts.migrate_embeddings --batch-size 128 --dry-run
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from neo4j import GraphDatabase

from Backend.config import settings
from Backend.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_SELECT_POSTS = """
    MATCH (p:Post)
    RETURN p.id AS id, coalesce(p.content, '') AS text
    ORDER BY p.id
    SKIP $skip
    LIMIT $limit
"""

_SELECT_MESSAGES = """
    MATCH (m:Message)
    RETURN m.id AS id, coalesce(m.text, '') AS text
    ORDER BY m.id
    SKIP $skip
    LIMIT $limit
"""

_SELECT_COMMENTS = """
    MATCH (c:Comment)
    RETURN c.id AS id, coalesce(c.content, '') AS text
    ORDER BY c.id
    SKIP $skip
    LIMIT $limit
"""

_UPDATE_POST_EMBEDDINGS = """
    UNWIND $rows AS row
    MATCH (p:Post {id: row.id})
    SET p.embedding = row.embedding
"""

_UPDATE_MESSAGE_EMBEDDINGS = """
    UNWIND $rows AS row
    MATCH (m:Message {id: row.id})
    SET m.embedding = row.embedding
"""

_UPDATE_COMMENT_EMBEDDINGS = """
    UNWIND $rows AS row
    MATCH (c:Comment {id: row.id})
    SET c.embedding = row.embedding
"""

_COUNT_POSTS = "MATCH (p:Post) RETURN count(p) AS n"
_COUNT_MESSAGES = "MATCH (m:Message) RETURN count(m) AS n"
_COUNT_COMMENTS = "MATCH (c:Comment) RETURN count(c) AS n"


@dataclass
class MigrationStats:
    posts: int = 0
    messages: int = 0
    comments: int = 0


def _count(session, cypher: str) -> int:
    result = session.run(cypher).single()
    return int(result["n"]) if result else 0


def _migrate_kind(
    *,
    session,
    embedder: EmbeddingService,
    select_cypher: str,
    update_cypher: str,
    batch_size: int,
    dry_run: bool,
    label: str,
) -> int:
    migrated = 0
    skip = 0

    while True:
        rows = session.run(select_cypher, skip=skip, limit=batch_size).data()
        if not rows:
            break

        texts = [str(r.get("text", "")) for r in rows]
        vectors = embedder.encode_batch(texts)

        payload = []
        for row, emb in zip(rows, vectors):
            payload.append({"id": str(row["id"]), "embedding": emb})

        if not dry_run:
            session.run(update_cypher, rows=payload)

        migrated += len(payload)
        skip += len(payload)
        logger.info("%s: migrated %d", label, migrated)

    return migrated


def _check_vector_indexes(session) -> None:
    """Check Neo4j version and warn if vector index dimensions might conflict."""
    try:
        # Get Neo4j version
        result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
        version_str = result.single()["version"]
        logger.info(f"Neo4j version detected: {version_str}")
        
        # Parse version (e.g. '5.12.0' -> (5, 12, 0))
        parts = version_str.split('-')[0].split('.')
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        
        if (major, minor) < (5, 15):
            logger.warning(
                "WARNING: Neo4j version is < 5.15. You likely need to MANUALLY DROP "
                "existing vector indexes before changing embedding dimension, "
                "as older versions do not cleanly allow altering index dimension."
            )
        else:
            logger.info("Neo4j version >= 5.15. Vector indexes may need to be dropped and recreated manually if dimensions changed.")
    except Exception as e:
        logger.warning(f"Could not verify Neo4j version or indexes: {e}")

def migrate_embeddings(batch_size: int = 128, dry_run: bool = False) -> MigrationStats:
    stats = MigrationStats()
    embedder = EmbeddingService()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    with driver.session(database=settings.neo4j_database) as session:
        _check_vector_indexes(session)
        total_posts = _count(session, _COUNT_POSTS)
        total_messages = _count(session, _COUNT_MESSAGES)
        total_comments = _count(session, _COUNT_COMMENTS)
        logger.info(
            "Counts: posts=%d messages=%d comments=%d",
            total_posts,
            total_messages,
            total_comments,
        )

        stats.posts = _migrate_kind(
            session=session,
            embedder=embedder,
            select_cypher=_SELECT_POSTS,
            update_cypher=_UPDATE_POST_EMBEDDINGS,
            batch_size=batch_size,
            dry_run=dry_run,
            label="Post",
        )
        stats.messages = _migrate_kind(
            session=session,
            embedder=embedder,
            select_cypher=_SELECT_MESSAGES,
            update_cypher=_UPDATE_MESSAGE_EMBEDDINGS,
            batch_size=batch_size,
            dry_run=dry_run,
            label="Message",
        )
        stats.comments = _migrate_kind(
            session=session,
            embedder=embedder,
            select_cypher=_SELECT_COMMENTS,
            update_cypher=_UPDATE_COMMENT_EMBEDDINGS,
            batch_size=batch_size,
            dry_run=dry_run,
            label="Comment",
        )

    driver.close()
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute Neo4j embeddings")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    stats = migrate_embeddings(batch_size=max(1, args.batch_size), dry_run=args.dry_run)
    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(
        f"{mode}: posts={stats.posts}, messages={stats.messages}, comments={stats.comments}"
    )


if __name__ == "__main__":
    main()
