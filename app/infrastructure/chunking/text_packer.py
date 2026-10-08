# chunking/text_packer.py
"""
Pack small text units into chunks of at most `max_tokens`.

A unit is an indivisible piece (a paragraph, a sentence, an equation
glued to the text below it, a reference...). Units are never split,
except when a single non-atomic unit is larger than `max_tokens`: it
is then cut into overlapping token windows.

Atomic units (those containing an equation) are never cut, even when
they exceed `max_tokens`, so LaTeX is never broken.

Overlap between two packed chunks is obtained by carrying over the
trailing units whose assembled size (separators included) fits in
`overlap_tokens`.

Budget accounting
-----------------
The tokens charged to the running chunk INCLUDE the separator that
would be inserted between the last unit and the candidate one, using
each unit's own `sep` when set, otherwise the default separator. This
is required for the "at most max_tokens" contract to hold on the text
that is actually assembled.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
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
    pages: Iterable[int],
    types: Iterable[ChunkType],
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


# ----------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------


def _separator_for(upcoming: Unit, default_sep: str) -> str:
    """Return the effective separator inserted before `upcoming`."""
    return upcoming.sep if upcoming.sep is not None else default_sep


def _separator_cost(upcoming: Unit, default_sep: str) -> int:
    """Token cost of the separator inserted before `upcoming`."""
    return count_tokens(_separator_for(upcoming, default_sep))


def _assembled_tokens(units: list[Unit], default_sep: str) -> int:
    """
    Exact token cost of joining `units` with the real separators.

    Uses the same join logic as `_join` so the count matches what will
    actually be stored in the chunk.
    """
    if not units:
        return 0

    return count_tokens(_join(units, default_sep))


def pack_units(
    units: list[Unit],
    config: ChunkingConfig,
    separator: str = "\n\n",
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current: list[Unit] = []
    current_tokens = 0
    has_new = False

    for unit in units:
        # Oversized unit: flush current, then split the unit on its own.
        if unit.tokens > config.max_tokens:
            if has_new:
                drafts.append(_to_draft(current, separator))

            current, current_tokens, has_new = [], 0, False
            drafts.extend(_split_oversized(unit, config))
            continue

        sep_cost = _separator_cost(unit, separator) if current else 0
        projected = current_tokens + sep_cost + unit.tokens

        if has_new and projected > config.max_tokens:
            drafts.append(_to_draft(current, separator))

            # Carry overlap, computed with the real separators.
            current = _overlap_tail(current, config.overlap_tokens, separator)
            current_tokens = _assembled_tokens(current, separator)

            # The overlap must leave room for the new unit; otherwise
            # drop it (the new unit is large enough already).
            sep_cost = _separator_cost(unit, separator) if current else 0
            if current_tokens + sep_cost + unit.tokens > config.max_tokens:
                current, current_tokens = [], 0

        sep_cost = _separator_cost(unit, separator) if current else 0
        current.append(unit)
        current_tokens += sep_cost + unit.tokens
        has_new = True

    if has_new:
        drafts.append(_to_draft(current, separator))

    return drafts


def _overlap_tail(
    units: list[Unit],
    overlap_tokens: int,
    default_sep: str,
) -> list[Unit]:
    """
    Return the trailing units whose assembled size fits in `overlap_tokens`.

    The cost charged is the exact assembled text (using each unit's own
    `sep`, falling back to `default_sep`), matching what `_join` will
    actually produce.
    """
    tail: list[Unit] = []

    for unit in reversed(units):
        candidate = [unit, *tail]
        if _assembled_tokens(candidate, default_sep) > overlap_tokens:
            break
        tail = candidate

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


def _join(units: list[Unit], default_sep: str) -> str:
    """Join units; a unit's own `sep` overrides the default separator."""
    text = units[0].text

    for unit in units[1:]:
        sep = unit.sep if unit.sep is not None else default_sep
        text += sep + unit.text

    return text


def _to_draft(units: list[Unit], separator: str) -> ChunkDraft:
    return ChunkDraft(
        text=_join(units, separator),
        types=order_types(t for unit in units for t in unit.types),
        pages=tuple(sorted({p for unit in units for p in unit.pages})),
    )