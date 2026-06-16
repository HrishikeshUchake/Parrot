"""OpenRouterClient – async wrapper around OpenRouter API using OpenAI SDK."""
from __future__ import annotations
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
import sys
from pathlib import Path

# Handle imports for both script and module usage
try:
    from config import settings
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Async client for OpenRouter remote LLM inference.

    Uses OpenRouter API (compatible with OpenAI SDK) for production-quality
    LLM responses. Supports multiple models via simple config change.

    See: https://openrouter.ai/docs
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        """Initialize OpenRouter/OpenAI client.

        If llm_provider is 'openai', this uses OpenAI endpoint/key.
        Otherwise it uses OpenRouter endpoint/key (backward compatibility).
        """
        provider = settings.llm_provider.lower() if settings.llm_provider else "openrouter"
        # Prefer OpenAI if key is present, even when llm_provider env is still openrouter.
        if settings.openai_api_key:
            provider = "openai"

        if provider == "openai":
            api_key = api_key or settings.openai_api_key
            model = model or settings.openai_model
            self._base_url = "https://api.openai.com/v1"
        else:
            api_key = api_key or settings.openrouter_api_key
            model = model or settings.openrouter_model
            self._base_url = "https://openrouter.ai/api/v1"

        if not api_key:
            raise ValueError(
                "API key is required. Set REMOTE_LLM_API_KEY or OPENROUTER_API_KEY environment variable."
            )

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def generate(self, prompt: str) -> str:
        """Generate response from anonymized prompt.

        Args:
            prompt: The prompt text (can be anonymized)

        Returns:
            Model response text
        """
        # Use httpx for async HTTP calls (matching ollama_client pattern)
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def _get_system_prompt(self) -> str:
        """Get the strict system prompt for token preservation.

        This prompt is designed to prevent the LLM from modifying, renumbering,
        or inventing anonymization tokens.
        """
        return """You are a helpful assistant that processes text containing anonymization placeholders.

CRITICAL RULES ABOUT PLACEHOLDERS:
====================================
1. PRESERVE EXACTLY: All placeholders like [EMAIL_ADDRESS_1], [PERSON_1], [PHONE_NUMBER_1], etc. MUST appear in your response EXACTLY as they appear in the input.

2. DO NOT MODIFY:
   - Do NOT renumber tokens (e.g., [EMAIL_ADDRESS_1] must stay [EMAIL_ADDRESS_1], never become [EMAIL_ADDRESS_2])
   - Do NOT delete tokens
   - Do NOT invent new tokens
   - Do NOT alter formatting of tokens (they must remain in brackets with underscore)

3. IF YOU REFERENCE A PLACEHOLDER:
   - Copy it EXACTLY as shown
   - Do not try to "improve" or "interpret" what it represents
   - It is a protected reference that will be restored later

4. COMPLIANCE CHECK:
   Before submitting your response, verify:
   ✓ All tokens from input are preserved with the exact same numbering
   ✓ No new tokens were created
   ✓ Token formatting is unchanged [TYPE_NUMBER]

Example:
INPUT: "Contact [PERSON_1] at [EMAIL_ADDRESS_1] for help with [PHONE_NUMBER_1]"
GOOD OUTPUT: "[PERSON_1] can be reached via [EMAIL_ADDRESS_1] or call [PHONE_NUMBER_1]"
BAD OUTPUT: "[PERSON_2] can be reached via email or phone" (WRONG - renumbered and deleted)

Your task is to be helpful while STRICTLY preserving all anonymization tokens."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def chat(self, messages: list[dict]) -> str:
        """Chat with message history.

        Automatically prepends the strict system prompt if not already present.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Model response text
        """
        # Ensure system prompt is included
        messages_with_system = messages.copy()
        if not messages_with_system or messages_with_system[0]["role"] != "system":
            messages_with_system.insert(0, {
                "role": "system",
                "content": self._get_system_prompt()
            })

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages_with_system,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def is_available(self) -> bool:
        """Synchronous health-check for OpenRouter API.

        Returns:
            True if API is accessible and API key is valid
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter health check failed: {e}")
            return False
