"""
Test: Real retrieval → Anonymize → OpenAI → Restore

Flow:
1. Take a real user query
2. Retrieve top 5 posts from the database using HybridSearchEngine
3. Format retrieved posts as context
4. Anonymize with Presidio
5. Send anonymized context to OpenAI
6. Restore PII tokens in response
7. Print final answer
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Backend.config import settings
from Backend.services.search_engine import HybridSearchEngine
from Backend.services.privacy import PresidioPrivatizer, NoOpPrivatizer
from Backend.llm.llm_provider import get_llm_provider


def _format_context(results) -> str:
    """Format retrieved search results into a readable context string."""
    if not results:
        return "No relevant posts found."
    lines = []
    for r in results:
        if r.result_type == "post" and r.post is not None:
            p = r.post
            snippet = p.content[:400] + ("..." if len(p.content) > 400 else "")
            author = p.account_acct or p.account_username or "unknown"
            engagement = (
                f"reblogs={p.reblogs_count}  favs={p.favourites_count}  "
                f"replies={p.replies_count}"
            )
            lines.append(
                f"[Status #{p.id}] Score={r.score:.2f} | @{author} | {engagement}\n"
                f"Tags={p.tags}\n"
                f"Content: {snippet}"
            )
        else:
            snippet = (r.content or "")[:400]
            lines.append(
                f"[{r.result_type} #{r.item_id}] Score={r.score:.2f}\n"
                f"Content: {snippet}"
            )
    return "\n---\n".join(lines)


async def test_retrieval_to_llm(user_query: str, top_k: int = 5):
    print("=" * 70)
    print("RETRIEVAL → ANONYMIZE → OPENAI → RESTORE")
    print("=" * 70)
    print(f"\nUser query: {user_query!r}")
    print(f"Top K: {top_k}")

    # ── Step 1: Retrieve top posts from database ──────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: RETRIEVING POSTS FROM DATABASE")
    print("─" * 70)

    engine = HybridSearchEngine()
    results = engine.search(query=user_query, top_k=top_k)

    if not results:
        print("❌ No results found in database. Make sure posts are ingested.")
        return

    print(f"✓ Retrieved {len(results)} posts")
    for i, r in enumerate(results, 1):
        if r.post:
            author = r.post.account_acct or r.post.account_username or "unknown"
            preview = r.post.content[:80].replace("\n", " ")
            print(f"  {i}. Score={r.score:.2f} | @{author} | {preview}...")
        else:
            print(f"  {i}. Score={r.score:.2f} | {r.result_type} #{r.item_id}")

    # ── Step 2: Format context ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: FORMATTING CONTEXT")
    print("─" * 70)
    context = _format_context(results)
    print(context[:600] + ("..." if len(context) > 600 else ""))

    # ── Step 3: Anonymize with Presidio ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: ANONYMIZING WITH PRESIDIO")
    print("─" * 70)

    if settings.privacy_enabled and settings.privacy_anonymizer.lower() == "presidio":
        try:
            privatizer = PresidioPrivatizer()
        except ImportError:
            print("⚠ Presidio not installed, using NoOp (no anonymization)")
            privatizer = NoOpPrivatizer()
    else:
        privatizer = NoOpPrivatizer()

    anonymized_context = privatizer.privatize_context(context)

    if anonymized_context != context:
        print("✓ PII detected and anonymized")
        # Show what was replaced
        if hasattr(privatizer, "get_current_mappings"):
            mappings = privatizer.get_current_mappings()
            for token, original in list(mappings.items())[:5]:
                print(f"  {token} → {original!r}")
            if len(mappings) > 5:
                print(f"  ... and {len(mappings) - 5} more")
    else:
        print("✓ No PII detected in retrieved posts")

    # ── Step 4: Send to OpenAI ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: SENDING ANONYMIZED CONTEXT TO OPENAI")
    print("─" * 70)

    prompt = f"""Based on the following social media posts, answer this question:

Question: {user_query}

Posts:
{anonymized_context}

Answer:"""

    llm_client = get_llm_provider()
    print("Calling OpenAI API...")

    try:
        llm_response = await llm_client.generate(prompt)
        print("✓ Response received from OpenAI")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return

    # ── Step 5: Restore tokens ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: RESTORING PII TOKENS")
    print("─" * 70)

    restored = privatizer.restore(llm_response)

    print("\n✅ FINAL ANSWER (with PII restored):")
    print("=" * 70)
    print(restored)
    print("=" * 70)


if __name__ == "__main__":
    # Change this query to test different topics
    query = sys.argv[1] if len(sys.argv) > 1 else "What are people saying about privacy?"
    asyncio.run(test_retrieval_to_llm(query, top_k=5))
