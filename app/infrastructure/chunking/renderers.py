# chunking/renderers.py
"""
Turn table / visual blocks into embeddable text.

Table contract
--------------
A table is rendered as one or more chunk texts. Parts respect
`max_tokens` when possible, but an individual row is NEVER split, so
a part containing a single oversized row may exceed the limit.

Every part repeats the caption (with a `(part i/n)` label), the
column header and the footnote, so each part is understandable on
its own.
"""
from __future__ import annotations

import logging

from app.domain.entities.block import TableBlock, VisualBlock
from app.infrastructure.chunking.token_counter import count_tokens

logger = logging.getLogger("paperlens.infrastructure.chunking")


def render_table(table: TableBlock) -> str:
    """Caption + Markdown grid + footnote (whole table, never split)."""
    parts = (table.caption, table.to_markdown(), table.footnote)

    return "\n\n".join(part for part in parts if part)


def render_table_parts(table: TableBlock, max_tokens: int) -> list[str]:
    """
    Render a table as one or more chunk texts.

    Rows are never split: each part contains a whole number of rows
    (possibly a single oversized one). When a single row exceeds
    `max_tokens`, that part exceeds the limit by design.
    """
    markdown = table.to_markdown()
    lines = markdown.splitlines() if markdown else []

    # Degenerate case: no header at all.
    if len(lines) < 2:
        caption = (table.caption or "").strip()
        footnote = (table.footnote or "").strip()
        joined = "\n\n".join(part for part in (caption, markdown, footnote) if part)
        return [joined] if joined else []

    header, rows = lines[:2], lines[2:]
    caption = table.caption or ""
    footnote = table.footnote or ""

    def assemble(row_lines: list[str], label: str = "") -> str:
        title = f"{caption} {label}".strip()
        grid = "\n".join(header + row_lines)
        return "\n\n".join(part for part in (title, grid, footnote) if part)

    # Whole table fits, or there is at most one row.
    if len(rows) <= 1 or count_tokens(assemble(rows)) <= max_tokens:
        return [assemble(rows)]

    # Budget left for data rows once the repeated caption/header/footnote
    # are accounted for.
    overhead = count_tokens(assemble([], "(part 99/99)"))
    budget = max(max_tokens - overhead, 1)

    groups: list[list[str]] = []
    current: list[str] = []
    used = 0

    for row in rows:
        cost = count_tokens(row) + 1  # +1 for the newline

        # A single oversized row: keep it alone, do not split it.
        if cost > budget and not current:
            groups.append([row])
            continue

        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0

        current.append(row)
        used += cost

    if current:
        groups.append(current)

    if len(groups) > 1:
        logger.info(
            "Table split into %d parts (max_tokens=%d)",
            len(groups),
            max_tokens,
        )

    return [
        assemble(group, f"(part {index}/{len(groups)})")
        for index, group in enumerate(groups, 1)
    ]


def render_visual(visual: VisualBlock) -> str:
    """Caption (+ footnote) of an image or a chart."""
    return "\n\n".join(part for part in (visual.caption, visual.footnote) if part)