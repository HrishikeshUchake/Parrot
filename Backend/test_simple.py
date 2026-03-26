"""
Simplest possible test - no async, no retry, just raw HTTP.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
import requests

print("=" * 70)
print("SIMPLE OPENROUTER TEST")
print("=" * 70)

api_key = settings.openrouter_api_key
model = settings.openrouter_model

print(f"\nAPI Key: {api_key[:40]}...")
print(f"Model: {model}")

# Make a simple request
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

data = {
    "model": model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.2,
    "max_tokens": 50,
}

print("\nMaking request...")
print(f"URL: {url}")
print(f"Headers: {headers}")
print(f"Data: {data}")

try:
    response = requests.post(url, headers=headers, json=data, timeout=10)

    print(f"\n✓ Got response!")
    print(f"Status: {response.status_code}")
    print(f"\nFull Response:")
    print(response.text)

    if response.status_code == 200:
        print("\n✅ SUCCESS!")
    else:
        print(f"\n❌ ERROR {response.status_code}")

except Exception as e:
    print(f"\n❌ Exception: {e}")
    print(f"Type: {type(e).__name__}")
