from __future__ import annotations

from pathlib import Path

from .hierarchy import detect_logical_level
from .utils import join_value


def to_block(
    item: dict,
    base_dir: Path,
) -> dict | None:
    kind = item.get("type", "unknown")

    page = item.get("page_idx")

    page = (
        page + 1
        if isinstance(page, int)
        else None
    )

    if kind == "text":
        text = (
            item.get("text") or ""
        ).strip()

        if text:
            return {
                "type": "text",
                "text": text,
                "page": page,
            }

        return None

    if kind == "equation":
        latex = (
            item.get("text") or ""
        ).strip()

        if latex:
            return {
                "type": "equation",
                "latex": latex,
                "page": page,
            }

        return None

    if kind == "table":
        block = {
            "type": "table",
            "caption": join_value(
                item.get("table_caption")
            ),
            "footnote": join_value(
                item.get("table_footnote")
            ),
            "page": page,
            "mineru_html": (
                item.get("table_body", "")
                or ""
            ),
        }

        if item.get("img_path"):
            block["image_path"] = str(
                (
                    base_dir / item["img_path"]
                ).resolve()
            )

        return block

    if kind == "image":
        image_path = item.get(
            "img_path",
            "",
        )

        return {
            "type": "image",
            "caption": join_value(
                item.get("image_caption")
            ),
            "footnote": join_value(
                item.get("image_footnote")
            ),
            "image_path": (
                str(
                    (
                        base_dir / image_path
                    ).resolve()
                )
                if image_path
                else ""
            ),
            "page": page,
        }

    block = {
        key: value
        for key, value in item.items()
        if key != "page_idx"
    }

    block["page"] = page

    return block


def get_section_context(
    stack: list[dict],
) -> dict:
    sections = [
        section
        for section in stack
        if section.get("level", 0) > 0
    ]

    if not sections:
        return {
            "section": None,
            "subsection": None,
            "section_path": [],
        }

    section = None
    subsection = None

    for current in sections:
        level = current.get("level", 0)

        if level == 1:
            section = current.get("title")
            subsection = None

        elif level >= 2:
            if section is None:
                section = current.get("title")
            else:
                subsection = current.get("title")

    section_path = [
        current.get("title")
        for current in sections
        if current.get("title")
    ]

    return {
        "section": section,
        "subsection": subsection,
        "section_path": section_path,
    }


def build_toc(root: dict) -> list[list]:
    toc = []

    def visit(section: dict) -> None:
        title = (
            section.get("title") or ""
        ).strip()

        level = section.get("level", 0)
        page = section.get("page")

        if title and level > 0:
            toc.append([
                level,
                title,
                page,
            ])

        for subsection in section.get(
            "subsections",
            [],
        ):
            visit(subsection)

    for section in root.get(
        "subsections",
        [],
    ):
        visit(section)

    return toc


def build_document(
    items: list,
    base_dir: Path,
    source: str,
) -> dict:
    root = {
        "title": None,
        "level": 0,
        "blocks": [],
        "subsections": [],
    }

    stack = [root]

    first_heading = None
    max_page = 0

    for item in items:
        if isinstance(
            item.get("page_idx"),
            int,
        ):
            max_page = max(
                max_page,
                item["page_idx"] + 1,
            )

        is_text = (
            item.get("type") == "text"
        )

        text = (
            item.get("text") or ""
        ).strip()

        mineru_level = item.get(
            "text_level"
        )

        if (
            is_text
            and text
            and mineru_level
        ):
            if first_heading is None:
                first_heading = text

            logical_level = detect_logical_level(
                text
            )

            page_index = item.get(
                "page_idx"
            )

            page = (
                page_index + 1
                if isinstance(page_index, int)
                else None
            )

            while (
                len(stack) > 1
                and stack[-1]["level"] >= logical_level
            ):
                stack.pop()

            section = {
                "title": text,
                "level": logical_level,
                "page": page,
                "blocks": [],
                "subsections": [],
            }

            stack[-1]["subsections"].append(
                section
            )

            stack.append(section)

            continue

        block = to_block(
            item,
            base_dir,
        )

        if block:
            context = get_section_context(
                stack
            )

            if context["section"] is not None:
                block["section"] = (
                    context["section"]
                )

            if context["subsection"] is not None:
                block["subsection"] = (
                    context["subsection"]
                )

            block["section_path"] = (
                context["section_path"]
            )

            stack[-1]["blocks"].append(
                block
            )

    return {
        "metadata": {
            "source_file": source,
            "pages": max_page,
            "title_guess": first_heading,
        },
        "toc": build_toc(root),
        "content": root,
    }


def iter_tables(section: dict):
    for block in section.get(
        "blocks",
        [],
    ):
        if block.get("type") == "table":
            yield block

    for subsection in section.get(
        "subsections",
        [],
    ):
        yield from iter_tables(subsection)


def count_blocks(
    section: dict,
    counts: dict | None = None,
) -> dict:
    if counts is None:
        counts = {}

    for block in section.get(
        "blocks",
        [],
    ):
        block_type = block.get(
            "type",
            "unknown",
        )

        counts[block_type] = (
            counts.get(block_type, 0) + 1
        )

    for subsection in section.get(
        "subsections",
        [],
    ):
        count_blocks(
            subsection,
            counts,
        )

    return counts