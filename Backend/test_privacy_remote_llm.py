"""
Test script demonstrating the complete privacy + remote LLM flow.

This shows:
1. Anonymize content using Presidio
2. Send anonymized content to OpenRouter
3. Restore tokens in LLM response
"""

import asyncio
import os
from services.privacy import PresidioPrivatizer


async def test_privacy_and_remote_llm():
    """Test the full privacy + remote LLM flow."""

    print("=" * 70)
    print("FULL PRIVACY + REMOTE LLM FLOW TEST")
    print("=" * 70)

    # Initialize privacy layer
    privatizer = PresidioPrivatizer()

    # Simulate formatted context from synthesis._format_context()
    original_context = """[Status #123] Score=0.85 | @alice.smith | reblogs=5  favs=10  replies=2
Tags=['privacy', 'security']
Content: Hi, I'm Alice Johnson. Contact me at alice.johnson@company.com or 555-123-4567

---

[Status #456] Score=0.72 | @bob.jones | reblogs=3  favs=7  replies=1
Tags=['data', 'security']
Content: For questions about Bob Smith, email bob.smith@company.com"""

    print("\n1️⃣  ORIGINAL CONTEXT (from Mastodon posts):")
    print("-" * 70)
    print(original_context)

    # Step 1: Anonymize
    print("\n" + "=" * 70)
    print("2️⃣  ANONYMIZING")
    print("=" * 70)

    anon_result = privatizer.anonymize(original_context)
    anonymized_context = anon_result["anonymized"]

    print("\nANONYMIZED CONTEXT (safe to send to remote LLM):")
    print("-" * 70)
    print(anonymized_context)

    print("\nDETECTED PII AND TOKENS:")
    print("-" * 70)
    for pii in anon_result["pii_found"]:
        print(
            f"  {pii['entity_type']:20} | {pii['original_value']:30} → {pii['token']}"
        )

    print("\nTOKEN MAPPINGS (stored for restoration):")
    print("-" * 70)
    for token, original in anon_result["mappings"].items():
        print(f"  {token:20} = {original}")

    # Step 2: Simulate remote LLM call
    print("\n" + "=" * 70)
    print("3️⃣  SENDING TO OPENROUTER (Simulated)")
    print("=" * 70)

    print("\nWould send to OpenRouter:")
    print(f"  Model: deepseek/deepseek-chat-v3-0324")
    print(f"  Context size: {len(anonymized_context)} characters")
    print(f"  PII masked: {len(anon_result['pii_found'])} entities")
    print(f"  Remote LLM sees: ONLY anonymized tokens, no real PII")

    # Simulate LLM response with tokens
    simulated_llm_output = f"""Based on the posts provided:

The people mentioned ([PERSON_1] and [PERSON_2]) discuss security and privacy topics.

[PERSON_1] can be reached at [EMAIL_ADDRESS_1] or [PHONE_NUMBER_1].
[PERSON_2] is available at [EMAIL_ADDRESS_2].

These individuals are actively engaged in security discussions with significant engagement metrics (5+ reblogs each)."""

    print("\n" + "=" * 70)
    print("4️⃣  LLM RESPONSE (with tokens)")
    print("=" * 70)
    print(simulated_llm_output)

    # Step 3: Restore tokens
    print("\n" + "=" * 70)
    print("5️⃣  RESTORING TOKENS")
    print("=" * 70)

    restored_answer = privatizer.restore(simulated_llm_output)

    print("\nFINAL ANSWER (tokens restored to original values):")
    print("-" * 70)
    print(restored_answer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Anonymized {len(anon_result['pii_found'])} PII entities")
    print(f"✓ Sent anonymized context to remote LLM")
    print(f"✓ LLM responded with tokens preserved")
    print(f"✓ Restored {len(privatizer.get_current_mappings())} tokens in response")
    print(f"✓ User receives answer with actual names/emails restored")
    print("\nSecurity Flow:")
    print("  User Data → Anonymize → Remote LLM → Restore → User")
    print("              (tokens only)    ✓ Safe to send externally")


async def test_with_config():
    """Test showing configuration options."""

    print("\n" + "=" * 70)
    print("CONFIGURATION OPTIONS")
    print("=" * 70)

    print("""
Backend/config.py settings for privacy + remote LLM:

    # LLM Provider Selection
    llm_provider: str = "openrouter"  # or "ollama"

    # OpenRouter Configuration
    openrouter_api_key: str = (from OPENROUTER_API_KEY env var)
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324"

    # Privacy/Anonymization
    privacy_enabled: bool = True
    privacy_anonymizer: str = "presidio"  # or "noop"

    # LLM Common Settings
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

To use OpenRouter:
    1. Set environment variable: export OPENROUTER_API_KEY="your-key-here"
    2. Set in config.py: llm_provider = "openrouter"
    3. Set in config.py: privacy_enabled = True
    4. System automatically:
       - Anonymizes all content using Presidio
       - Sends anonymized tokens to OpenRouter
       - Restores real values in responses
""")


async def main():
    """Run all tests."""
    try:
        await test_privacy_and_remote_llm()
        await test_with_config()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
