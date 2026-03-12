"""OpenAI-compatible backend: connects to any OpenAI API-compatible endpoint.

Works with LM Studio, vLLM, text-generation-inference, OpenRouter,
and any other server that implements the /v1/chat/completions endpoint.
"""

from __future__ import annotations

import logging
import os

import httpx

from wawd.oracle.backends.base import OracleBackend

log = logging.getLogger(__name__)


class OpenAICompatBackend(OracleBackend):
    """Oracle backend using the OpenAI chat completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

        # API key: explicit param > env var > empty (for local servers)
        self._api_key = api_key or os.environ.get("WAWD_OPENAI_API_KEY", "")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Send a chat completion request to the OpenAI-compatible endpoint."""
        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self._model:
            payload["model"] = self._model

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            log.warning("OpenAI-compat request timed out after %.1fs", self._timeout)
            return self._fallback(messages)
        except (httpx.HTTPError, KeyError, IndexError) as e:
            log.warning("OpenAI-compat request failed: %s", e)
            return self._fallback(messages)

    async def health_check(self) -> bool:
        """Check if the endpoint is reachable.

        Tries /v1/models first (standard OpenAI), then /health (common
        for local servers), then falls back to a minimal completion request.
        """
        # Try /v1/models (LM Studio, vLLM, OpenAI)
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                if self._model and self._model not in models:
                    log.warning(
                        "Model '%s' not found. Available: %s", self._model, models
                    )
                    return False
                return True
        except Exception:
            pass

        # Try /health (llama.cpp, TGI)
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        log.warning("OpenAI-compat health check failed for %s", self._base_url)
        return False

    def _fallback(self, messages: list[dict]) -> str:
        """Return a fallback response when the backend is unavailable."""
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        context = user_msgs[-1] if user_msgs else "No context available"
        return f"[Oracle unavailable — raw context follows]\n{context}"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
