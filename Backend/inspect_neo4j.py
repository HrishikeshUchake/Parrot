"""
inspect_neo4j.py
================
Quick sanity-check script for the Neo4j graph store.

Run from repo root:
    python -m Backend.inspect_neo4j
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

# Allow running as  python -m Backend.inspect_neo4j  OR  python Backend/inspect_neo4j.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from Backend.config import settings


def _count(session, label: str) -> int:
    """Return the number of nodes with *label*, safely handling an empty graph."""
    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
    return result["c"] if result else 0


def _count_rel(session, rel_type: str) -> int:
    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").single()
    return result["c"] if result else 0


def main() -> None:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    db = settings.neo4j_database

    with driver.session(database=db) as session:
        # Node counts – separate queries so a missing label never returns None
        posts    = _count(session, "Post")
        tags     = _count(session, "Tag")
        accounts = _count(session, "Account")
        print(f"Nodes  →  Posts: {posts}  Tags: {tags}  Accounts: {accounts}")

        # Relationship counts
        has_tag  = _count_rel(session, "HAS_TAG")
        authored = _count_rel(session, "AUTHORED")
        print(f"Edges  →  HAS_TAG: {has_tag}  AUTHORED: {authored}")

        # Sample 3 posts
        print("\nSample statuses:")
        sample = session.run("""
            MATCH (p:Post)
            RETURN p.id               AS id,
                   p.account_acct     AS account_acct,
                   p.tags_json        AS tags_json,
                   p.reblogs_count    AS reblogs_count,
                   p.favourites_count AS favourites_count,
                   p.language         AS language,
                   size(p.embedding)  AS emb_dim
            LIMIT 3
        """).data()
        for row in sample:
            tags = json.loads(row["tags_json"]) if row["tags_json"] else []
            print(
                f"  Status {row['id']}  @{row['account_acct']}  "
                f"reblogs={row['reblogs_count']}  favs={row['favourites_count']}  "
                f"lang={row['language']}  embedding_dim={row['emb_dim']}\n"
                f"    tags={tags}"
            )

        # Tag frequency
        print("\nTop 10 hashtags:")
        top_tags = session.run("""
            MATCH (t:Tag)<-[:HAS_TAG]-(p:Post)
            RETURN t.name AS tag, count(p) AS post_count
            ORDER BY post_count DESC LIMIT 10
        """).data()
        for t in top_tags:
            print(f"  #{t['tag']}  ({t['post_count']} posts)")

        # Vector index status
        print("\nVector indexes:")
        indexes = session.run("""
            SHOW INDEXES
            WHERE type = 'VECTOR'
        """).data()
        for idx in indexes:
            print(f"  name={idx.get('name')}  state={idx.get('state')}  "
                  f"entity={idx.get('labelsOrTypes')}  property={idx.get('properties')}")

    driver.close()


if __name__ == "__main__":
    main()
