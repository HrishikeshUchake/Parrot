import logging
import os
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
        self.model = settings.openai_model
        self._client = AsyncClient(timeout=Timeout(15.0))

    async def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for remote synthesis mode")

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
            logger.error("Remote LLM failed: %s", e)
            raise


def get_llm_provider() -> LLMProvider:
    if settings.llm_backend.lower() == "openai":
        return OpenAIProvider()
    return OllamaClient()


def get_llm_provider_for_backend(backend: str | None = None) -> LLMProvider:
    """Return provider for a specific backend, defaulting to global setting."""
    selected = (backend or settings.llm_backend).strip().lower()
    if selected == "openai":
        return OpenAIProvider()
    return OllamaClient()


def get_node_llm_provider(node_name: str) -> LLMProvider:
    """Return provider for a node-specific backend setting."""
    node_key = (node_name or "").strip().lower()
    backend_map = {
        "query_analyzer": settings.analyzer_llm_backend,
        "analyzer": settings.analyzer_llm_backend,
        "router": settings.router_llm_backend,
        "synthesis": settings.synthesis_llm_backend,
    }
    env_var_map = {
        "query_analyzer": "ANALYZER_LLM_BACKEND",
        "analyzer": "ANALYZER_LLM_BACKEND",
        "router": "ROUTER_LLM_BACKEND",
        "synthesis": "SYNTHESIS_LLM_BACKEND",
    }

    backend = backend_map.get(node_key, settings.llm_backend)

    # If a node-specific backend wasn't explicitly set in environment,
    # inherit the global backend to avoid surprising defaults.
    node_env_var = env_var_map.get(node_key)
    if node_env_var and not os.getenv(node_env_var):
        backend = settings.llm_backend

    return get_llm_provider_for_backend(backend)
