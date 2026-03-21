import logging
from typing import Protocol

from httpx import AsyncClient, Timeout
from ..config import settings
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def generate(self, prompt: str, system: str = "") -> str:
        ...


class OpenAIProvider:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.ollama_model  # Reuse or add new config
        self._client = AsyncClient(timeout=Timeout(15.0))
        self._fallback = OllamaClient()

    async def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            logger.warning("OpenAI API key not set, falling back to Ollama")
            return await self._fallback.generate(prompt, system=system)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model,
                "messages": [],
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            }
            if system:
                payload["messages"].append(
                    {"role": "system", "content": system})
            payload["messages"].append({"role": "user", "content": prompt})

            url = f"{self.base_url.rstrip('/')}/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"

            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(
                f"Remote LLM failed: {e}. Falling back to local Ollama.")
            return await self._fallback.generate(prompt, system=system)


def get_llm_provider() -> LLMProvider:
    if settings.llm_backend.lower() == "openai":
        return OpenAIProvider()
    return OllamaClient()
