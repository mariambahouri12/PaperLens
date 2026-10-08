# chunking/token_counter.py
"""
Token counting and token-based splitting helpers.

`tiktoken` is a REQUIRED dependency: the chunker enforces a strict
`max_tokens` budget, which is only meaningful with a real tokenizer.
If `tiktoken` is not installed, importing this module raises the
usual `ImportError`, which is the desired behaviour (fail loudly).
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoder():
    """Return the cached `cl100k_base` encoder (a module-level singleton)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact number of tokens in `text`."""
    if not text:
        return 0

    return len(_encoder().encode(text))


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
    tokens = encoder.encode(text)

    return [
        encoder.decode(tokens[start:end])
        for start, end in _windows(len(tokens), size, overlap)
    ]