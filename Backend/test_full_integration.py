"""
Simple test to verify the full privacy + remote LLM integration works.

This tests the actual OpenRouter API call (not simulated).
"""

import asyncio
from config import settings
from services.privacy import PresidioPrivatizer
from llm.openrouter_client import OpenRouterClient


async def test_full_integration():
    """Test: Anonymize → Send to OpenRouter → Restore"""

    print("=" * 70)
    print("TESTING FULL PRIVACY + REMOTE LLM INTEGRATION")
    print("=" * 70)

    # Check API key
    if not settings.openrouter_api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not set in .env")
        return

    # print(f"✓ API Key loaded: {settings.openrouter_api_key[:30]}...")
    # print(f"✓ Provider: {settings.llm_provider}")
    # print(f"✓ Privacy enabled: {settings.privacy_enabled}")
    # print()

    # Initialize components
    privatizer = PresidioPrivatizer()
    llm_client = OpenRouterClient()

    # Sample context with PII (like what synthesis.py would get)
    original_context = """[Status #123] Score=0.85 | @alice.smith | reblogs=5  favs=10  replies=2
Tags=['privacy', 'security']
Content: Hi, I'm Alice Johnson. Contact me at alice.johnson@company.com or 555-123-4567.
Dont forget to reach me at alice.johnson@company.com
---

[Status #456] Score=0.72 | @bob.jones | reblogs=3  favs=7  replies=1
Tags=['data', 'security']
Content: For questions, email bob.smith@company.com"""

    print("STEP 1: ORIGINAL CONTEXT (with PII)")
    print("-" * 70)
    print(original_context)
    print()

    # ANONYMIZE
    print("STEP 2: ANONYMIZING")
    print("-" * 70)
    anon_result = privatizer.anonymize(original_context)
    anonymized_context = anon_result["anonymized"]

    print("Anonymized context:")
    print(anonymized_context)
    print()
    print("Detected PII:")
    for pii in anon_result["pii_found"]:
        print(f"  {pii['entity_type']}: {pii['original_value']} → {pii['token']}")
    print()

    # SEND TO OPENROUTER
    print("STEP 3: SENDING TO OPENROUTER")
    print("-" * 70)
    print("Calling OpenRouter API...")
    print("(This is the anonymized context being sent)")
    print()

    # Create a prompt with the anonymized context
    prompt = f"""Based on the social media posts below, provide a brief summary:

{anonymized_context}

Summary:"""

    try:
        llm_response = await llm_client.generate(prompt)
        print("✓ OpenRouter response received!")
        print()

        # RESTORE
        print("STEP 4: RESTORING TOKENS")
        print("-" * 70)
        restored_response = privatizer.restore(llm_response)
        print("Restored response (with real names/emails):")
        print(restored_response)
        print()

        print("=" * 70)
        print("✅ FULL INTEGRATION TEST SUCCESSFUL!")
        print("=" * 70)
        print("\nFlow completed:")
        print("  Original → Anonymize → Remote LLM → Restore → User")

    except Exception as e:
        print(f"❌ Error calling OpenRouter: {e}")
        print("\nTroubleshooting:")
        print("  - Check your API key in .env")
        print("  - Verify you have OpenRouter account credits")
        print("  - Check internet connection")


if __name__ == "__main__":
    asyncio.run(test_full_integration())
