"""
Test showing the improved system prompt for token preservation.

This demonstrates how the stronger system prompt helps the remote LLM
preserve anonymization tokens correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm.openrouter_client import OpenRouterClient

print("=" * 80)
print("SYSTEM PROMPT FOR TOKEN PRESERVATION")
print("=" * 80)

client = OpenRouterClient()

print("\nSTRONG SYSTEM PROMPT (Now in use):")
print("-" * 80)
print(client._get_system_prompt())

print("\n" + "=" * 80)
print("KEY FEATURES OF THE STRONG PROMPT")
print("=" * 80)

features = [
    ("1. EXPLICIT RULES",
     "Lists exactly what tokens are and what NOT to do"),

    ("2. DO NOT MODIFY LIST",
     "Explicitly forbids:\n     - Renumbering tokens\n     - Deleting tokens\n     - Inventing tokens\n     - Altering formatting"),

    ("3. COMPLIANCE CHECK",
     "Includes a pre-submit verification checklist"),

    ("4. CONCRETE EXAMPLES",
     "Shows GOOD output vs BAD output"),

    ("5. PRIORITY FRAMING",
     "Makes token preservation the primary concern"),
]

for title, explanation in features:
    print(f"\n✓ {title}")
    print(f"  → {explanation}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

old_prompt = (
    "Do not modify placeholders like [EMAIL_ADDRESS_1], [PERSON_1], etc. "
    "Keep them exactly as-is in your response."
)

new_prompt_snippet = """CRITICAL RULES ABOUT PLACEHOLDERS:
- DO NOT renumber tokens
- DO NOT delete tokens
- DO NOT invent new tokens
- DO NOT alter formatting

COMPLIANCE CHECK:
✓ All tokens from input preserved with exact numbering
✓ No new tokens created
✓ Token formatting unchanged [TYPE_NUMBER]

Example: [PERSON_1] at [EMAIL_ADDRESS_1] → exact reproduction required"""

print(f"\nOLD PROMPT (weak):\n  {old_prompt}")
print(f"\nNEW PROMPT (strong):\n  {new_prompt_snippet}")

print("\n" + "=" * 80)
print("WHY THIS MATTERS")
print("=" * 80)

scenarios = [
    {
        "issue": "Renumbering",
        "example": "[EMAIL_ADDRESS_1] becomes [EMAIL_ADDRESS_2]",
        "protection": "Explicit rule + example showing this is WRONG",
    },
    {
        "issue": "Deletion",
        "example": "LLM drops [PHONE_NUMBER_1] from response",
        "protection": "Compliance checklist reminds to preserve ALL tokens",
    },
    {
        "issue": "Invention",
        "example": "LLM creates [EMAIL_ADDRESS_99] that doesn't exist",
        "protection": "Explicit rule: 'Do NOT invent new tokens'",
    },
    {
        "issue": "Interpretation",
        "example": "LLM tries to understand what [PERSON_1] means",
        "protection": "'Do not try to improve or interpret' rule",
    },
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{i}. {scenario['issue'].upper()}")
    print(f"   Problem: {scenario['example']}")
    print(f"   Protection: {scenario['protection']}")

print("\n" + "=" * 80)
print("INTEGRATION")
print("=" * 80)

print("""
When synthesis.py calls OpenRouter:

1. Content is anonymized by PresidioPrivatizer
   └─ [PERSON_1], [EMAIL_ADDRESS_1], etc.

2. Strong system prompt is automatically included
   └─ Tells the model to NEVER modify tokens

3. Model processes with strict instructions
   └─ Model knows this is critical

4. Response returned with tokens preserved
   └─ Token validation checks for integrity

5. Tokens restored to original values
   └─ User sees real names/emails
""")

print("\n" + "=" * 80)
print("TESTING THE PROMPT")
print("=" * 80)

print("""
To test with real API calls, run:

  python test_full_integration.py

This will:
  ✓ Anonymize sample posts
  ✓ Send with strong system prompt to OpenRouter
  ✓ Validate token integrity in response
  ✓ Restore and show final answer

The strong prompt significantly reduces the chance of token tampering.
""")
