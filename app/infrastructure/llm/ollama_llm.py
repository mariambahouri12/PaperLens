"""
Local LLM adapter for Ollama.

Requires Ollama running locally and the model pulled, e.g.:

    ollama pull qwen3:8b

All generation parameters come from Settings; nothing is hard-coded.

Qwen3 is a reasoning model. For RAG the reasoning is switched off by
default (`llm_think=False`): answers are faster, cheaper in context tokens
and the answer never gets cut by `num_ctx` in the middle of the reasoning.
Any <think>...</think> block that still reaches the answer is removed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import ollama

from app.config.settings import Settings
from app.domain.exceptions import LLMError
from app.domain.repositories.llm import LLMPort

logger = logging.getLogger("paperlens.infrastructure.llm")

_THINK_BLOCK = re.compile(
    r"<think>.*?(?:</think>|$)\s*",
    re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove Qwen thinking blocks, including an unfinished final block."""
    return _THINK_BLOCK.sub("", text or "").strip()


class OllamaLLM(LLMPort):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = ollama.Client()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": self.settings.temperature,
                    "num_ctx": self.settings.num_ctx,
                },
                think=self.settings.llm_think,
            )
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

        answer = _strip_thinking(self._content(response))

        if not answer:
            raise LLMError(
                "Ollama returned an empty answer (reasoning not finished? "
                "check num_ctx and llm_think)"
            )

        return answer

    @staticmethod
    def _content(response: Any) -> str:
        """Extract generated text from supported Ollama response shapes."""

        if isinstance(response, dict):
            try:
                return response["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise LLMError(
                    f"Unexpected Ollama response shape: {exc}"
                ) from exc

        try:
            return response.message.content
        except AttributeError as exc:
            raise LLMError(
                f"Unexpected Ollama response shape: {exc}"
            ) from exc
