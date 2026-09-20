"""
Local LLM adapter for Ollama.

Requires Ollama running locally and the model pulled, e.g.:

    ollama pull qwen3:8b

All generation parameters come from Settings; nothing is hard-coded.
"""
from __future__ import annotations

import logging

import ollama

from app.config.settings import Settings
from app.domain.exceptions import LLMError
from app.domain.repositories.llm import LLMPort

logger = logging.getLogger("paperlens.infrastructure.llm")


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
            )
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

        try:
            return response["message"]["content"]
        except Exception:
            try:
                return response.message.content  # type: ignore[attr-defined]
            except Exception as exc:
                raise LLMError(f"Unexpected Ollama response shape: {exc}") from exc