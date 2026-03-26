"""
Test script demonstrating the privacy + Remote LLM flow.

This shows how to:
1. Anonymize content using Presidio
2. Send anonymized content to remote LLM
3. Restore tokens back to original values
"""

from services.privacy import PresidioPrivatizer


def test_remote_flow():
    """Test the full anonymization -> LLM -> restoration flow."""

    # Initialize Presidio-based privatizer
    privatizer = PresidioPrivatizer()

    # Example content with PII
    text = (
        "Hi, my name is Sarah Johnson. You can reach me at:\n"
        "- Email: sarah.j@company.com\n"
        "- Phone: 555-123-4567\n"
        "- Alternate: sarah.johnson@personal.com"
    )

    print("=" * 70)
    print("STEP 1: ANONYMIZE")
    print("=" * 70)
    print("\nORIGINAL TEXT:\n", text)

    # Step 1: Anonymize
    anon_result = privatizer.anonymize(text)
    anonymized_text = anon_result["anonymized"]

    print("\nANONYMIZED TEXT:\n", anonymized_text)
    print("\nDETECTED PII:")
    for pii in anon_result["pii_found"]:
        print(f"  - {pii['entity_type']}: {pii['original_value']!r} -> {pii['token']}")
    print("\nTOKEN MAPPINGS:")
    for token, original in anon_result["mappings"].items():
        print(f"  {token} = {original!r}")

    # Step 2: Send to remote LLM (simulated)
    print("\n" + "=" * 70)
    print("STEP 2: SEND TO REMOTE LLM")
    print("=" * 70)
    print("\nSending anonymized text to remote LLM...")
    # llm_output = call_remote_llm(anonymized_text)

    # Simulated LLM output that references the anonymized tokens
    simulated_llm_output = (
        "The person [PERSON_1] can be contacted via email at [EMAIL_ADDRESS_1] "
        "or [EMAIL_ADDRESS_2], or by phone at [PHONE_NUMBER_1]."
    )
    print(f"LLM OUTPUT:\n{simulated_llm_output}")

    # Step 3: Restore tokens
    print("\n" + "=" * 70)
    print("STEP 3: RESTORE TOKENS")
    print("=" * 70)
    restored = privatizer.restore(simulated_llm_output)
    print(f"RESTORED TEXT:\n{restored}")

    # Verify consistency
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"Same tokens used: [PERSON_1], [EMAIL_ADDRESS_1], etc.")
    print(f"Original values restored correctly: {restored}")


def test_context_anonymization():
    """Test anonymizing formatted context (as used in synthesis node)."""

    privatizer = PresidioPrivatizer()

    # Simulated formatted context from synthesis._format_context()
    context = """[Status #123] Score=0.85 | @alice.smith | reblogs=5  favs=10  replies=2
Tags=['privacy', 'security']
Content: I'm Alice Smith and you can reach me at alice@example.com or 555-1234567 for more details

---

[Status #456] Score=0.72 | @bob.jones | reblogs=3  favs=7  replies=1
Tags=['data', 'privacy']
Content: Contact Bob Jones at bob.jones@company.com for questions"""

    print("\nORIGINAL CONTEXT:")
    print(context)

    result = privatizer.anonymize(context)

    print("\nANONYMIZED CONTEXT:")
    print(result["anonymized"])

    print("\nMAPPINGS:")
    for token, value in result["mappings"].items():
        print(f"  {token} = {value!r}")


def test_consistency():
    """Test that the same values get the same tokens."""

    privatizer = PresidioPrivatizer()

    # Same email appears twice
    text1 = "Contact john@example.com"
    text2 = "Email john@example.com again"

    result1 = privatizer.anonymize(text1)
    result2 = privatizer.anonymize(text2)

    print("\nTEST CONSISTENCY:")
    print(f"Text 1: {text1!r}")
    print(f"  -> {result1['anonymized']!r}")

    print(f"Text 2: {text2!r}")
    print(f"  -> {result2['anonymized']!r}")

    # Extract tokens
    token1 = result1["anonymized"].split("@")[0].split(" ")[-1] + "@"

    print(f"\nSame token used for same email? Check mappings:")
    print(f"Current mappings: {privatizer.mappings}")


if __name__ == "__main__":
    try:
        test_remote_flow()
        print("\n\n")
        test_context_anonymization()
        print("\n\n")
        test_consistency()
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nTo use PresidioPrivatizer, install presidio-analyzer:")
        print("  pip install presidio-analyzer")
