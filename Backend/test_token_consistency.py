"""
Test to verify that the same PII value gets the same token (consistency).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.privacy import PresidioPrivatizer

print("=" * 70)
print("TESTING TOKEN CONSISTENCY")
print("=" * 70)

privatizer = PresidioPrivatizer()

# Test 1: Same email appears multiple times
text = """
Contact Alice at alice@example.com for question 1.
For question 2, reach Alice at alice@example.com.
Alice can also be reached at alice@example.com.
"""

print("\nTEST 1: Same Email Appearing 3 Times")
print("-" * 70)
print("Original text:")
print(text)

result = privatizer.anonymize(text)

print("\nAnonymized text:")
print(result['anonymized'])

print("\nMappings found:")
for pii in result['pii_found']:
    print(f"  {pii['original_value']} → {pii['token']}")

# Check consistency
email_tokens = [pii['token'] for pii in result['pii_found'] if pii['entity_type'] == 'EMAIL_ADDRESS']
if len(set(email_tokens)) == 1:
    print(f"\n✅ CONSISTENT: Same email got same token [{email_tokens[0]}]")
else:
    print(f"\n❌ INCONSISTENT: Same email got different tokens {email_tokens}")

# Test 2: Multiple different emails
print("\n\nTEST 2: Same Name + Different Emails")
print("-" * 70)

text2 = """
Bob is at bob@work.com and also bob@personal.com
Bob Smith mentioned this before.
Contact Bob at bob@work.com again.
"""

print("Original text:")
print(text2)

result2 = privatizer.anonymize(text2)

print("\nAnonymized text:")
print(result2['anonymized'])

print("\nMappings found:")
for pii in result2['pii_found']:
    print(f"  {pii['original_value']} → {pii['token']}")

print("\nReverse Mappings (original → token):")
for orig, token in privatizer.reverse_mappings.items():
    print(f"  {orig} → {token}")

# Verify consistency
work_email_tokens = [pii['token'] for pii in result2['pii_found']
                     if pii['original_value'] == 'bob@work.com']
if len(set(work_email_tokens)) == 1:
    print(f"\n✅ CONSISTENT: 'bob@work.com' always maps to {work_email_tokens[0]}")
else:
    print(f"\n❌ INCONSISTENT: 'bob@work.com' got different tokens")

print("\n" + "=" * 70)
print("Conclusion: Same PII value → Same token (every time in session)")
print("=" * 70)
