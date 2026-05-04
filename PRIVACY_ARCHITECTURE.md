# Privacy & Anonymization Architecture with Presidio

## Overview

The privacy layer uses **Microsoft Presidio** to detect and anonymize PII (Personally Identifiable Information) before sending content to remote LLMs.

Key concepts:
- **Anonymize**: Replace detected PII with tokens (e.g., `[PERSON_1]`, `[EMAIL_ADDRESS_1]`)
- **Mappings**: Store token → original value for later restoration
- **Restore**: Replace tokens back to original values after LLM processing
- **Consistent tokens**: Same PII value always maps to same token within a session

---

## Data Flow: Anonymization → Remote LLM → Restoration

```
Original Text/Context
  ├─ "Contact Sarah Johnson at sarah@example.com or 555-1234567"
    ↓
PresidioPrivatizer.anonymize()
  ├─ Presidio detects: PERSON, EMAIL_ADDRESS, PHONE_NUMBER
  ├─ Replace with tokens: [PERSON_1], [EMAIL_ADDRESS_1], [PHONE_NUMBER_1]
  ├─ Store mappings:
  │    "[PERSON_1]" → "Sarah Johnson"
  │    "[EMAIL_ADDRESS_1]" → "sarah@example.com"
  │    "[PHONE_NUMBER_1]" → "555-1234567"
    ↓
Anonymized Text
  ├─ "Contact [PERSON_1] at [EMAIL_ADDRESS_1] or [PHONE_NUMBER_1]"
    ↓
Send to Remote LLM (OpenRouter, Claude, etc.)
    ↓
LLM Processing & Output
  ├─ "You can reach [PERSON_1] by email at [EMAIL_ADDRESS_1]"
    ↓
PresidioPrivatizer.restore()
  ├─ Replace tokens with original values from mappings
    ↓
Final Output
  └─ "You can reach Sarah Johnson by email at sarah@example.com"
```

---

## PII Detection Types (Presidio)

Presidio can detect:
- `EMAIL_ADDRESS` - Email addresses
- `PHONE_NUMBER` - Phone numbers
- `PERSON` - Person names
- `CREDIT_CARD` - Credit card numbers
- `LOCATION` - Geographic locations
- `IP_ADDRESS` - IP addresses
- `MEDICAL_LICENSE` - Medical license numbers
- And more...

---

## Integration with Synthesis Node

### Updated Flow in `Backend/agents/synthesis.py`

```python
from services.privacy import PresidioPrivatizer

# Initialize once at module level
_privatizer = PresidioPrivatizer()

async def synthesis_node(state: AgentState) -> dict:
    """Generate final answer with privacy protection."""
    results = state.get("search_results", [])

    # Step 1: Format context (original behavior)
    context = _format_context(results)

    # Step 2: ANONYMIZE before sending to LLM
    anonymized_context = _privatizer.privatize_context(context)

    # Step 3: Create prompt with anonymized context
    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        context=anonymized_context  # ← Uses anonymized version
    )

    # Step 4: Call remote LLM with anonymized content
    answer = await _client.generate(prompt)

    # Step 5: RESTORE any token references in the answer
    restored_answer = _privatizer.restore(answer)

    return {
        "answer": restored_answer,  # ← Return restored text to user
        "reasoning": f"Route: {state.get('route')} | Results: {len(results)}",
    }
```

---

## Usage Examples

### Basic Anonymization & Restoration

```python
from services.privacy import PresidioPrivatizer

privatizer = PresidioPrivatizer()

# Anonymize text
original = "Hi Sarah, email me at sarah.j@company.com"
result = privatizer.anonymize(original)

print(result['anonymized'])
# Output: "Hi [PERSON_1], email me at [EMAIL_ADDRESS_1]"

print(result['mappings'])
# Output: {'[PERSON_1]': 'Sarah', '[EMAIL_ADDRESS_1]': 'sarah.j@company.com'}

# Restore
llm_output = "You can reach [PERSON_1] at [EMAIL_ADDRESS_1]"
restored = privatizer.restore(llm_output)
print(restored)
# Output: "You can reach Sarah at sarah.j@company.com"
```

### PII Detection Details

```python
result = privatizer.anonymize("My name is John and my email is john@example.com")

for pii in result['pii_found']:
    print(f"- {pii['entity_type']}: '{pii['original_value']}' → {pii['token']}")
    print(f"  Confidence: {pii['confidence']:.2f}")

# Output:
# - PERSON: 'John' → [PERSON_1]
#   Confidence: 0.85
# - EMAIL_ADDRESS: 'john@example.com' → [EMAIL_ADDRESS_1]
#   Confidence: 0.99
```

### Component-Level Privacy

```python
# If you want finer control, use methods individually:
content = "Contact Alice at alice@corp.com"
anonymized_content = privatizer.privatize_content(content)

username = "bob_smith"
acct = "bob.smith@company.com"
anon_username, anon_acct = privatizer.privatize_author(username, acct)

tags = ["sarah", "team-meeting", "#personal"]
anon_tags = privatizer.privatize_tags(tags)
```

### Session Management

```python
privatizer = PresidioPrivatizer()

# Session 1: Process some content
result1 = privatizer.anonymize("Email john@example.com")
# [EMAIL_ADDRESS_1] → john@example.com

# Same content in session: consistent token
result2 = privatizer.anonymize("Contact john@example.com again")
# john@example.com → [EMAIL_ADDRESS_1] (same token!)

# Get current mappings
print(privatizer.get_current_mappings())

# Start new session
privatizer.reset()
# Now john@example.com will map to [EMAIL_ADDRESS_1] again or different number
```

---

## Configuration

### In `Backend/config.py`

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Privacy/Anonymization
    privacy_enabled: bool = True
    privacy_anonymizer: str = "presidio"  # "presidio" or "noop"
```

### Using Configuration

```python
from config import settings
from services.privacy import PresidioPrivatizer, NoOpPrivatizer

if settings.privacy_enabled:
    if settings.privacy_anonymizer == "presidio":
        privatizer = PresidioPrivatizer()
    else:
        privatizer = NoOpPrivatizer()  # No-op for testing
else:
    privatizer = NoOpPrivatizer()
```

---

## Testing

Run the included test script:

```bash
cd Backend
python test_privacy_flow.py
```

This demonstrates:
1. Anonymizing text with multiple PII types
2. Formatting and anonymizing context (as used in synthesis)
3. Simulated remote LLM call
4. Restoration of tokens
5. Token consistency across multiple references

---

## Installation

Install Presidio analyzer:

```bash
pip install presidio-analyzer
```

Optional: For additional anonymization capabilities:

```bash
pip install presidio-anonymizer
```

---

## Important Notes

### Token Consistency

⚠️ Tokens are **consistent within a session** (one `PresidioPrivatizer` instance).

- Same PII value → same token
- Example: `"john@example.com"` always becomes `[EMAIL_ADDRESS_1]` in the same session
- Different sessions: tokens reset (call `reset()`)

### Stateless APIs

For stateless request-based APIs (FastAPI, Flask):

```python
# Option 1: Create new privatizer per request
@app.post("/query")
async def query(request: QueryRequest):
    privatizer = PresidioPrivatizer()  # Fresh per request
    # ...

# Option 2: Store in session context
@app.post("/query")
async def query(request: QueryRequest, session: dict):
    if 'privatizer' not in session:
        session['privatizer'] = PresidioPrivatizer()
    privatizer = session['privatizer']
    # ...
```

### Performance Considerations

- Presidio's AnalyzerEngine runs entity detection on each call
- For large batches, consider batching anonymization calls
- Analyzer is stateless and can be reused

---

## Next Steps

1. ✅ Create `Backend/services/privacy.py` with `PresidioPrivatizer`
2. ✅ Create test script `Backend/test_privacy_flow.py`
3. Update `Backend/config.py` with privacy settings
4. Modify `Backend/agents/synthesis.py` to use privatizer
5. Test end-to-end with real Mastodon posts
6. Integrate with remote LLM service (OpenRouter, Claude, etc.)
