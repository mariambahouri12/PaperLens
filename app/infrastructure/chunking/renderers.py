"""Turn table / visual blocks into embeddable text."""
from __future__ import annotations

from app.domain.entities.block import TableBlock, VisualBlock


def render_table(table: TableBlock) -> str:
    """Caption + Markdown grid + footnote."""
    parts = (table.caption, table.to_markdown(), table.footnote)

    return "\n\n".join(part for part in parts if part)


def render_visual(visual: VisualBlock) -> str:
    """Caption (+ footnote) of an image or a chart."""
    return "\n\n".join(part for part in (visual.caption, visual.footnote) if part)