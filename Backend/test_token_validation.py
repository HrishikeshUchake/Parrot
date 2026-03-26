"""
Test token integrity validation after remote LLM processing.

Shows how to detect if the LLM modified, renumbered, or invented tokens.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.privacy import PresidioPrivatizer

print("=" * 80)
print("TESTING TOKEN INTEGRITY VALIDATION")
print("=" * 80)

privatizer = PresidioPrivatizer()

# Step 1: Anonymize content
original = """
Contact Alice at alice@example.com or 555-1234567.
Also reach Alice at alice@example.com for help.
Bob is at bob@work.com.
"""

print("\nSTEP 1: ANONYMIZE CONTENT")
print("-" * 80)
print("Original:")
print(original)

result = privatizer.anonymize(original)
anonymized = result["anonymized"]

print("\nAnonymized:")
print(anonymized)

print("\nMappings:")
for token, original_val in privatizer.mappings.items():
    print(f"  {token} = {original_val}")

# Step 2: Simulate different LLM responses with different token issues
print("\n" + "=" * 80)
print("STEP 2: TEST LLM RESPONSES WITH DIFFERENT TOKEN ISSUES")
print("=" * 80)

test_cases = [
    {
        "name": "✅ GOOD: LLM preserves all tokens correctly",
        "response": f"You can reach [PERSON_1] at [EMAIL_ADDRESS_1] or [PHONE_NUMBER_1]. "
                   f"Also try [EMAIL_ADDRESS_2] for [PERSON_2].",
        "expected_valid": True,
    },
    {
        "name": "❌ RENUMBERED: LLM changed token numbers",
        "response": "You can reach [PERSON_1] at [EMAIL_ADDRESS_2] or [PHONE_NUMBER_2]. "
                   "Also try [EMAIL_ADDRESS_3] for [PERSON_2].",
        "expected_valid": False,
    },
    {
        "name": "❌ INVENTED: LLM created new tokens",
        "response": "You can reach [PERSON_1] at [EMAIL_ADDRESS_1]. "
                   "New contact: [EMAIL_ADDRESS_99] for [PERSON_99].",
        "expected_valid": False,
    },
    {
        "name": "❌ DELETED: LLM removed some tokens",
        "response": "You can reach [PERSON_1] at [EMAIL_ADDRESS_1]. "
                   "No phone number mentioned.",
        "expected_valid": False,
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n\nTEST CASE {i}: {test_case['name']}")
    print("-" * 80)
    print(f"LLM Response:\n  {test_case['response']}")

    # Validate tokens
    validation = privatizer.validate_tokens(test_case["response"])

    print(f"\nValidation Result:")
    print(f"  Is Valid: {validation['is_valid']}")
    print(f"  Found Tokens: {validation['found_tokens']}")
    print(f"  Unknown Tokens: {validation['unknown_tokens']}")
    print(f"  Missing Tokens: {validation['missing_tokens']}")

    if validation["warnings"]:
        print(f"\n  ⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"    - {warning}")

    # Check if matches expectation
    if validation['is_valid'] == test_case['expected_valid']:
        print(f"\n  ✅ CORRECT: Validation result matches expected")
    else:
        print(f"\n  ❌ MISMATCH: Expected valid={test_case['expected_valid']}, "
              f"got {validation['is_valid']}")

# Step 3: Demonstrate restore with validation
print("\n" + "=" * 80)
print("STEP 3: RESTORE WITH VALIDATION (strict & non-strict)")
print("=" * 80)

good_response = (
    "You can reach [PERSON_1] at [EMAIL_ADDRESS_1] or [PHONE_NUMBER_1]. "
    "Also try [EMAIL_ADDRESS_2] for [PERSON_2]."
)

bad_response = (
    "You can reach [PERSON_1] at [EMAIL_ADDRESS_99]. "
    "Call [PHONE_999]."
)

print("\n\n3a) GOOD RESPONSE (all tokens valid)")
print("-" * 80)
print(f"LLM Response:\n  {good_response}")

restored, validation = privatizer.restore_with_validation(good_response, strict=False)
print(f"\nRestored (strict=False):\n  {restored}")
print(f"Validation: {validation['is_valid']}")

print("\n\n3b) BAD RESPONSE (invalid tokens, strict=False)")
print("-" * 80)
print(f"LLM Response:\n  {bad_response}")

try:
    restored, validation = privatizer.restore_with_validation(bad_response, strict=False)
    print(f"\nRestored (strict=False):\n  {restored}")
    print(f"Validation: {validation['is_valid']}")
    print(f"⚠️  Warnings were logged but restoration continued")
except ValueError as e:
    print(f"Error: {e}")

print("\n\n3c) BAD RESPONSE (invalid tokens, strict=True)")
print("-" * 80)
print(f"LLM Response:\n  {bad_response}")

try:
    restored, validation = privatizer.restore_with_validation(bad_response, strict=True)
    print(f"Restored:\n  {restored}")
except ValueError as e:
    print(f"✅ CAUGHT ERROR (strict mode):")
    print(f"  {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Token Integrity Validation protects against:
  ✓ Renumbered tokens - LLM changed [EMAIL_1] to [EMAIL_2]
  ✓ Invented tokens - LLM created [EMAIL_99] that doesn't exist
  ✓ Missing tokens - LLM deleted mentions of [PHONE_NUMBER_1]
  ✓ Malformed tokens - LLM altered token formatting

Two restore modes:
  • restore_with_validation(strict=False) - Log warnings, continue anyway
  • restore_with_validation(strict=True) - Fail if any token is invalid
""")
