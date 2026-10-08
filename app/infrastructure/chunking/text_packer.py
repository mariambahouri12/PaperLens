"""
Pack small text units into chunks of at most `max_tokens`.

A unit is an indivisible piece (a paragraph, a sentence, an equation glued
to the text below it, a reference...). Units are never split, except when a
single non-atomic unit is larger than max_tokens: it is then cut into
overlapping token windows.

Atomic units (those containing an equation) are never cut, even when they
exceed max_tokens, so LaTeX is never broken.

Overlap between two packed chunks is obtained by carrying over the
trailing units that fit in `overlap_tokens`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.entities.chunk import ChunkType
from app.infrastructure.chunking.config import ChunkingConfig
from app.infrastructure.chunking.drafts import ChunkDraft, order_types
from app.infrastructure.chunking.token_counter import count_tokens, split_by_tokens

logger = logging.getLogger("paperlens.infrastructure.chunking")


@dataclass(frozen=True)
class Unit:
    text: str
    tokens: int
    pages: tuple[int, ...]
    types: tuple[ChunkType, ...]
    sep: str | None = None  
    atomic: bool = False  # never cut, even if larger than max_tokens


def make_unit(
    text: str,
    pages,
    types,
    sep: str | None = None,
    atomic: bool = False,
) -> Unit:
    return Unit(
        text=text,
        tokens=count_tokens(text),
        pages=tuple(sorted({page for page in pages if page})),
        types=order_types(types),
        sep=sep,
        atomic=atomic,
    )


def pack_units(
    units: list[Unit],
    config: ChunkingConfig,
    separator: str = "\n\n",
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current: list[Unit] = []
    current_tokens = 0
    has_new = False  # True if `current` holds units not emitted yet

    for unit in units:
        if unit.tokens > config.max_tokens:
            if has_new:
                drafts.append(_to_draft(current, separator))

            current, current_tokens, has_new = [], 0, False
            drafts.extend(_split_oversized(unit, config))
            continue

        if has_new and current_tokens + unit.tokens > config.max_tokens:
            drafts.append(_to_draft(current, separator))

            current = _overlap_tail(current, config.overlap_tokens)
            current_tokens = sum(item.tokens for item in current)

            # The carried overlap must leave room for the new unit.
            if current_tokens + unit.tokens > config.max_tokens:
                current, current_tokens = [], 0

        current.append(unit)
        current_tokens += unit.tokens
        has_new = True

    if has_new:
        drafts.append(_to_draft(current, separator))

    return drafts


def _overlap_tail(units: list[Unit], overlap_tokens: int) -> list[Unit]:
    tail: list[Unit] = []
    total = 0

    for unit in reversed(units):
        if total + unit.tokens > overlap_tokens:
            break

        tail.insert(0, unit)
        total += unit.tokens

    return tail


def _split_oversized(unit: Unit, config: ChunkingConfig) -> list[ChunkDraft]:
    if unit.atomic:
        logger.warning(
            "Unit containing an equation (%d tokens) exceeds max_tokens "
            "and is kept whole",
            unit.tokens,
        )
        return [ChunkDraft(text=unit.text, types=unit.types, pages=unit.pages)]

    windows = split_by_tokens(unit.text, config.max_tokens, config.overlap_tokens)

    return [
        ChunkDraft(text=window, types=unit.types, pages=unit.pages)
        for window in windows
    ]


def _join(units: list[Unit], separator: str) -> str:
    """Join units; a unit's own `sep` overrides the default separator."""
    text = units[0].text

    for unit in units[1:]:
        text += (separator if unit.sep is None else unit.sep) + unit.text

    return text


def _to_draft(units: list[Unit], separator: str) -> ChunkDraft:
    return ChunkDraft(
        text=_join(units, separator),
        types=order_types(t for unit in units for t in unit.types),
        pages=tuple(sorted({p for unit in units for p in unit.pages})),
    )