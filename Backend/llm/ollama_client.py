"""OllamaClient async wrapper around the Ollama REST API."""
from __future__ import annotations
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class OllamaClient:
    """Thin async client for local Ollama inference."""

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def generate(self, prompt: str, system: str = "") -> str:
        """Non-streaming generation. Returns the model response text."""
        if system:
            return await self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def chat(self, messages: list[dict]) -> str:
        """Chat endpoint (system + user messages)."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    def is_available(self) -> bool:
        """Synchronous health-check."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self._base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
