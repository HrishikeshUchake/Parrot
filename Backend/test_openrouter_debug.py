"""
Debug test to see detailed OpenRouter API error message.
"""

import asyncio
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings


async def test_openrouter_api():
    """Test OpenRouter API with detailed error output."""

    print("=" * 70)
    print("TESTING OPENROUTER API CONNECTION")
    print("=" * 70)

    provider = settings.llm_provider.lower() if settings.llm_provider else "openrouter"
    # Prefer OpenAI if OPENAI_API_KEY is available.
    if settings.openai_api_key:
        provider = "openai"

    if provider == "openai":
        api_key = settings.openai_api_key
        model = settings.openai_model
        base_url = "https://api.openai.com/v1"
    else:
        api_key = settings.openrouter_api_key
        model = settings.openrouter_model
        base_url = "https://openrouter.ai/api/v1"

    print(f"\n✓ API Key: {api_key[:30]}...")
    print(f"✓ Model: {model}")
    print(f"✓ Base URL: {base_url}")

    # Test the API
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("\nSending test request to OpenRouter...")

            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'merhaba' in mongolian "
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 100,
                },
            )

            print(f"\nStatus Code: {resp.status_code}")
            print(f"Response Headers: {dict(resp.headers)}")
            print(f"\nResponse Body:")
            print(resp.text)

            if resp.status_code == 200:
                print("\n✅ SUCCESS!")
                result = resp.json()
                print(f"Answer: {result['choices'][0]['message']['content']}")
            else:
                print(f"\n❌ ERROR {resp.status_code}")
                try:
                    error = resp.json()
                    print(f"Error Details: {error}")
                except:
                    print(f"Response: {resp.text}")

        except Exception as e:
            print(f"\n❌ Connection Error: {e}")
            print(f"Type: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(test_openrouter_api())
