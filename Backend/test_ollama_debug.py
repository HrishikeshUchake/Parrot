"""
Debug test for local Ollama connectivity and generation.

Run:
    cd /Users/leyla/Desktop/parrot/Backend
    python test_ollama_debug.py
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path

# Allow importing `Backend.*` when run from /Backend directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backend.config import settings
from Backend.llm.ollama_client import OllamaClient


async def test_ollama() -> None:
    print("=" * 70)
    print("TESTING OLLAMA CONNECTION")
    print("=" * 70)

    print(f"\nBase URL: {settings.ollama_base_url}")
    print(f"Model:    {settings.ollama_model}")

    client = OllamaClient()

    print("\n1) Health check (/api/tags)")
    available = client.is_available()
    print(f"   Available: {available}")

    if not available:
        print("\n❌ Ollama is not reachable.")
        print("Start it with: ollama serve")
        print(f"And make sure model exists: ollama pull {settings.ollama_model}")
        return

    print("\n2) Generation check")
    prompt = "Reply with exactly one word: Hello"

    try:
        answer = await client.generate(prompt)
        print("✅ Generation succeeded")
        print(f"Answer: {answer}")
    except Exception as exc:
        print(f"❌ Generation failed: {exc}")
        print("If model is missing, run:")
        print(f"  ollama pull {settings.ollama_model}")


if __name__ == "__main__":
    asyncio.run(test_ollama())
