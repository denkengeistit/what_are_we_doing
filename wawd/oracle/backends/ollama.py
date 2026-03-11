"""OllamaBackend: connects to a local Ollama instance (typically in Docker)."""

from __future__ import annotations

import logging

import httpx

from wawd.oracle.backends.base import OracleBackend

log = logging.getLogger(__name__)


class OllamaBackend(OracleBackend):
    """Oracle backend using the Ollama REST API."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Send a chat completion request to Ollama."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except httpx.TimeoutException:
            log.warning("Ollama request timed out after %.1fs", self._timeout)
            return self._fallback(messages)
        except (httpx.HTTPError, KeyError) as e:
            log.warning("Ollama request failed: %s", e)
            return self._fallback(messages)

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check for exact match or match without tag
            for m in models:
                if m == self._model or m.split(":")[0] == self._model.split(":")[0]:
                    return True
            log.warning("Model '%s' not found. Available: %s", self._model, models)
            return False
        except (httpx.HTTPError, Exception) as e:
            log.warning("Ollama health check failed: %s", e)
            return False

    def _fallback(self, messages: list[dict]) -> str:
        """Return a fallback response when the oracle is unavailable."""
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        context = user_msgs[-1] if user_msgs else "No context available"
        return f"[Oracle unavailable — raw context follows]\n{context}"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
