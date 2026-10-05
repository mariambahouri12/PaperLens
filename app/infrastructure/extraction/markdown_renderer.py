from __future__ import annotations

from utils import join_value


def cell_to_markdown(value) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def table_to_markdown(
    block: dict,
) -> list[str]:
    lines = []

    caption = (
        block.get("caption") or ""
    ).strip()

    if caption:
        lines.extend([
            f"**{caption}**",
            "",
        ])

    columns = block.get(
        "columns",
        [],
    )

    rows = block.get(
        "rows",
        [],
    )

    if columns:
        lines.append(
            "| "
            + " | ".join(
                cell_to_markdown(column)
                for column in columns
            )
            + " |"
        )

        lines.append(
            "| "
            + " | ".join(
                "---"
                for _ in columns
            )
            + " |"
        )

        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    cell_to_markdown(cell)
                    for cell in row
                )
                + " |"
            )

        lines.append("")

    footnote = (
        block.get("footnote") or ""
    ).strip()

    if footnote:
        lines.extend([
            f"*{footnote}*",
            "",
        ])

    return lines


def block_to_markdown(
    block: dict,
) -> list[str]:
    block_type = block.get("type")

    if block_type == "table":
        return table_to_markdown(block)

    if block_type == "equation":
        latex = (
            block.get("latex") or ""
        ).strip()

        if not latex:
            return []

        return [
            (
                latex
                if latex.startswith("$$")
                else f"$$\n{latex}\n$$"
            ),
            "",
        ]

    if block_type == "list":
        items = (
            block.get("list_items")
            or []
        )

        if items:
            return [
                *(str(item) for item in items),
                "",
            ]

    if block_type == "image":
        caption = (
            block.get("caption") or ""
        ).strip()

        return (
            [f"**{caption}**", ""]
            if caption
            else []
        )

    if block_type == "chart":
        caption = join_value(
            block.get("chart_caption")
        )

        return (
            [f"**{caption}**", ""]
            if caption
            else []
        )

    text = (
        block.get("text") or ""
    ).strip()

    return (
        [text, ""]
        if text
        else []
    )


def section_to_markdown(
    section: dict,
    lines: list[str],
) -> None:
    title = (
        section.get("title") or ""
    ).strip()

    if title:
        level = min(
            int(
                section.get(
                    "level",
                    1,
                )
            ),
            6,
        )

        lines.extend([
            "#" * level + " " + title,
            "",
        ])

    for block in section.get(
        "blocks",
        [],
    ):
        lines.extend(
            block_to_markdown(block)
        )

    for subsection in section.get(
        "subsections",
        [],
    ):
        section_to_markdown(
            subsection,
            lines,
        )


def document_to_markdown(
    document: dict,
) -> str:
    lines: list[str] = []

    section_to_markdown(
        document["content"],
        lines,
    )

    result = "\n".join(lines)

    while "\n\n\n" in result:
        result = result.replace(
            "\n\n\n",
            "\n\n",
        )

    return result.strip()