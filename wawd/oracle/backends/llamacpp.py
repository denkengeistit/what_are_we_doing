"""LlamaCppBackend: connects to a llama.cpp server with OpenAI-compatible API."""

from __future__ import annotations

import logging

import httpx

from wawd.oracle.backends.base import OracleBackend

log = logging.getLogger(__name__)


class LlamaCppBackend(OracleBackend):
    """Oracle backend using the llama.cpp OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Send a chat completion request to llama.cpp."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            log.warning("llama.cpp request timed out after %.1fs", self._timeout)
            return self._fallback(messages)
        except (httpx.HTTPError, KeyError, IndexError) as e:
            log.warning("llama.cpp request failed: %s", e)
            return self._fallback(messages)

    async def health_check(self) -> bool:
        """Check if the llama.cpp server is reachable."""
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            resp.raise_for_status()
            return True
        except (httpx.HTTPError, Exception) as e:
            log.warning("llama.cpp health check failed: %s", e)
            return False

    def _fallback(self, messages: list[dict]) -> str:
        """Return a fallback response when the backend is unavailable."""
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        context = user_msgs[-1] if user_msgs else "No context available"
        return f"[Oracle unavailable — raw context follows]\n{context}"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
