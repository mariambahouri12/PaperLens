"""
Local LLM adapter for Ollama.

Requires Ollama running locally and the model pulled, e.g.:

    ollama pull qwen3:8b

All generation parameters come from Settings; nothing is hard-coded.

Qwen3 is a reasoning model. For RAG the reasoning is switched off by
default (`llm_think=False`): answers are faster, cheaper in context tokens
and the answer never gets cut by `num_ctx` in the middle of the reasoning.
Any <think>...</think> block that still reaches the answer is removed.

After every call the adapter stores Ollama's own statistics in
`last_stats`; they are logged by the application layer (QueryTrace).
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

_NS_PER_SECOND = 1_000_000_000


def _strip_thinking(text: str) -> str:
    """Remove Qwen thinking blocks, including an unfinished final block."""
    return _THINK_BLOCK.sub("", text or "").strip()


def _field(response: Any, name: str, default: Any = 0) -> Any:
    """Read a field from a dict-like or attribute-style Ollama response."""
    if isinstance(response, dict):
        value = response.get(name)
    else:
        value = getattr(response, name, None)

    return default if value is None else value


class OllamaLLM(LLMPort):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = ollama.Client()
        self.last_stats: dict[str, Any] | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        options: dict[str, Any] = {
            "temperature": self.settings.temperature,
            "num_ctx": self.settings.num_ctx,
        }

        # Cap the output length so generation can never run away.
        num_predict = getattr(self.settings, "llm_num_predict", None)
        if num_predict:
            options["num_predict"] = num_predict

        try:
            response = self._client.chat(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options=options,
                think=self.settings.llm_think,
                # Keep the model loaded between questions.
                keep_alive=getattr(self.settings, "llm_keep_alive", "30m"),
            )
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

        self.last_stats = self._collect_stats(response)

        answer = _strip_thinking(self._content(response))

        if not answer:
            raise LLMError(
                "Ollama returned an empty answer (reasoning not finished? "
                "check num_ctx and llm_think)"
            )

        return answer

    @staticmethod
    def _collect_stats(response: Any) -> dict[str, Any]:
        """Extract Ollama's timing statistics from a chat response."""
        gen_tokens = _field(response, "eval_count")
        gen_time = _field(response, "eval_duration") / _NS_PER_SECOND

        return {
            "total_s": _field(response, "total_duration") / _NS_PER_SECOND,
            "load_s": _field(response, "load_duration") / _NS_PER_SECOND,
            "prompt_tokens": _field(response, "prompt_eval_count"),
            "prompt_s": _field(response, "prompt_eval_duration") / _NS_PER_SECOND,
            "gen_tokens": gen_tokens,
            "gen_s": gen_time,
            "tokens_per_s": gen_tokens / gen_time if gen_time else 0.0,
        }

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