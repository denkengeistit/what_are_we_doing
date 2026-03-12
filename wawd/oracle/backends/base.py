"""Abstract base class for oracle SLM backends."""

from abc import ABC, abstractmethod


class OracleBackend(ABC):
    """Base class for SLM backends (Ollama, llama.cpp, etc.)."""

    @abstractmethod
    async def generate(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Send messages to the SLM and return the response text."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and a model is loaded."""
        ...

    async def close(self) -> None:
        """Clean up resources. Override if the backend holds connections."""
        pass
