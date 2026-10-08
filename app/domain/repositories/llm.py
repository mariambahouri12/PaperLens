# domain/repositories/llm.py
"""Port: generate text with a local LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMPort(ABC):
    """
    Local LLM capable of instruction-following generation.

    The port is synchronous: RAG use cases run on a worker thread if
    the caller needs async. Errors must be raised as LLMError, never
    swallowed silently.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant's answer as plain text.

        Raises LLMError if the model is unavailable or returns an empty
        answer (e.g. reasoning cut by num_ctx).
        """

    def close(self) -> None:
        """Release any resource held by the client. No-op by default."""