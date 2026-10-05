"""
Token counting and token-based splitting helpers.

Uses tiktoken when available (cl100k_base, a reasonable proxy for any
modern LLM tokenizer); falls back to a character-based estimate
(~4 chars/token) otherwise. Never raises.
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

_CHARS_PER_TOKEN = 4


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

    encoder = _encoder()

    if encoder is None:
        return max(1, len(text) // _CHARS_PER_TOKEN)

    return len(encoder.encode(text))


def _windows(length: int, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) index pairs of overlapping windows."""
    step = size - overlap
    start = 0

    while True:
        end = min(start + size, length)
        yield start, end

        if end >= length:
            return

        start += step


def split_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` tokens with `overlap` tokens shared."""
    if size <= overlap:
        raise ValueError("size must be greater than overlap")

    encoder = _encoder()

    if encoder is None:
        return [
            text[start:end]
            for start, end in _windows(
                len(text),
                size * _CHARS_PER_TOKEN,
                overlap * _CHARS_PER_TOKEN,
            )
        ]

    tokens = encoder.encode(text)

    return [
        encoder.decode(tokens[start:end])
        for start, end in _windows(len(tokens), size, overlap)
    ]