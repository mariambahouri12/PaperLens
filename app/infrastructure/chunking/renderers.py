"""Turn table / visual blocks into embeddable text."""
from __future__ import annotations

from app.domain.entities.block import TableBlock, VisualBlock
from app.infrastructure.chunking.token_counter import count_tokens


def render_table(table: TableBlock) -> str:
    """Caption + Markdown grid + footnote (whole table, never split)."""
    parts = (table.caption, table.to_markdown(), table.footnote)

    return "\n\n".join(part for part in parts if part)


def render_table_parts(table: TableBlock, max_tokens: int) -> list[str]:
    """
    Render a table as one or more chunk texts of at most `max_tokens`.

    Tables are only cut between rows, never inside one. Every part repeats
    the caption (with a "(part i/n)" label), the column header and the
    footnote, so each part is understandable on its own.
    """
    lines = table.to_markdown().splitlines()
    header, rows = lines[:2], lines[2:]  # header row + "| --- |" separator row
    caption = table.caption or ""
    footnote = table.footnote or ""

    def assemble(row_lines: list[str], label: str = "") -> str:
        title = f"{caption} {label}".strip()
        grid = "\n".join(header + row_lines)

        return "\n\n".join(part for part in (title, grid, footnote) if part)

    if len(rows) <= 1 or count_tokens(assemble(rows)) <= max_tokens:
        return [assemble(rows)]

    # Tokens left for data rows once the repeated caption / header / footnote
    # are accounted for.
    overhead = count_tokens(assemble([], "(part 99/99)"))
    budget = max(max_tokens - overhead, 1)

    groups: list[list[str]] = []
    current: list[str] = []
    used = 0

    for row in rows:
        cost = count_tokens(row) + 1  # +1 for the newline

        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0

        current.append(row)
        used += cost

    groups.append(current)

    return [
        assemble(group, f"(part {index}/{len(groups)})")
        for index, group in enumerate(groups, 1)
    ]


def render_visual(visual: VisualBlock) -> str:
    """Caption (+ footnote) of an image or a chart."""
    return "\n\n".join(part for part in (visual.caption, visual.footnote) if part)