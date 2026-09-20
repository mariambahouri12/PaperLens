"""
Token counting helper.

Uses tiktoken when available (cl100k_base, a reasonable proxy for any
modern LLM tokenizer); falls back to a character-based estimate
(~4 chars/token) otherwise. Never raises.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))