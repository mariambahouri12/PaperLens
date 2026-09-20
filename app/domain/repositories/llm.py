"""Port: generate text with a local LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...