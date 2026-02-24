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


def main() -> None:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    db = settings.neo4j_database

    with driver.session(database=db) as session:
        # Node counts
        counts = session.run("""
            MATCH (p:Post)  WITH count(p) AS posts
            MATCH (t:Tag)   WITH posts, count(t) AS tags
            MATCH (u:User)  RETURN posts, tags, count(u) AS users
        """).single()
        print(f"Nodes  →  Posts: {counts['posts']}  Tags: {counts['tags']}  Users: {counts['users']}")

        # Relationship counts
        rels = session.run("""
            MATCH ()-[r:HAS_TAG]->()  WITH count(r) AS ht
            MATCH ()-[r:AUTHORED]->() RETURN ht, count(r) AS au
        """).single()
        print(f"Edges  →  HAS_TAG: {rels['ht']}  AUTHORED: {rels['au']}")

        # Sample 3 posts
        print("\nSample posts:")
        sample = session.run("""
            MATCH (p:Post)
            RETURN p.id AS id, p.title AS title, p.tags_json AS tags_json,
                   p.views AS views, p.user_id AS user_id,
                   size(p.embedding) AS emb_dim
            LIMIT 3
        """).data()
        for row in sample:
            tags = json.loads(row["tags_json"]) if row["tags_json"] else []
            print(
                f"  Post #{row['id']:>3}  views={row['views']:>5}  "
                f"embedding_dim={row['emb_dim']}  "
                f"tags={tags}\n"
                f"    {row['title']}"
            )

        # Tag frequency
        print("\nTop 10 tags:")
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
