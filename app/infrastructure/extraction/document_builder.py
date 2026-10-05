from __future__ import annotations

from pathlib import Path

from front_matter import (
    ABSTRACT_TITLE,
    AUTHORS_TITLE,
    split_abstract,
)
from hierarchy import detect_logical_level, has_section_number
from utils import join_value


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def page_number(item: dict) -> int | None:
    """Convert MinerU's 0-based page index to a 1-based page number."""
    page = item.get("page_idx")
    return page + 1 if isinstance(page, int) else None


def resolve_path(base_dir: Path, relative: str) -> str:
    """Resolve a MinerU relative asset path to an absolute path."""
    return str((base_dir / relative).resolve()) if relative else ""


def new_section(
    title: str | None,
    level: int,
    page: int | None,
) -> dict:
    return {
        "title": title,
        "level": level,
        "page": page,
        "blocks": [],
        "subsections": [],
    }


# ----------------------------------------------------------------------
# Items -> blocks
# ----------------------------------------------------------------------


def to_block(item: dict, base_dir: Path) -> dict | None:
    """Convert a MinerU content item into a normalized block."""
    kind = item.get("type", "unknown")
    page = page_number(item)

    if kind == "text":
        text = (item.get("text") or "").strip()

        if text:
            return {"type": "text", "text": text, "page": page}

        return None

    if kind == "equation":
        latex = (item.get("text") or "").strip()

        if latex:
            return {"type": "equation", "latex": latex, "page": page}

        return None

    if kind == "table":
        block = {
            "type": "table",
            "caption": join_value(item.get("table_caption")),
            "footnote": join_value(item.get("table_footnote")),
            "page": page,
            "mineru_html": item.get("table_body", "") or "",
        }

        if item.get("img_path"):
            block["image_path"] = resolve_path(base_dir, item["img_path"])

        return block

    if kind == "image":
        return {
            "type": "image",
            "caption": join_value(item.get("image_caption")),
            "footnote": join_value(item.get("image_footnote")),
            "image_path": resolve_path(
                base_dir,
                item.get("img_path", ""),
            ),
            "page": page,
        }

    # Any other type (list, chart, aside_text, ...): keep all fields as-is.
    block = {
        key: value
        for key, value in item.items()
        if key != "page_idx"
    }

    block["page"] = page

    return block


# ----------------------------------------------------------------------
# Section context / TOC
# ----------------------------------------------------------------------


def get_section_context(stack: list[dict]) -> dict:
    """Derive section, subsection and full path from the section stack."""
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
    """Flatten the section tree into [level, title, page] entries."""
    toc = []

    def visit(section: dict) -> None:
        title = (section.get("title") or "").strip()
        level = section.get("level", 0)
        page = section.get("page")

        if title and level > 0:
            toc.append([level, title, page])

        for subsection in section.get("subsections", []):
            visit(subsection)

    for section in root.get("subsections", []):
        visit(section)

    return toc


# ----------------------------------------------------------------------
# Tree builder
# ----------------------------------------------------------------------


class _TreeBuilder:
    """
    Builds the section tree from MinerU's content_list.json.

    Handles the "front matter": after the paper title, any text that comes
    before the abstract is stored in an "Authors" section, and a paragraph
    starting with "Abstract" opens an "ABSTRACT" section.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.root = new_section(None, 0, None)
        self.stack = [self.root]
        self.max_page = 0
        self.title_section: dict | None = None
        self.in_front_matter = True

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, item: dict) -> None:
        """Process one content item (heading or block)."""
        page = item.get("page_idx")

        if isinstance(page, int):
            self.max_page = max(self.max_page, page + 1)

        text = (item.get("text") or "").strip()

        if item.get("type") == "text" and text and item.get("text_level"):
            self._add_heading(text, page_number(item))
            return

        block = to_block(item, self.base_dir)

        if block is not None:
            self._add_block(block)

    def to_document(self, source: str) -> dict:
        """Return the final document: metadata, TOC and content tree."""
        title = self.title_section["title"] if self.title_section else None

        return {
            "metadata": {
                "source_file": source,
                "pages": self.max_page,
                "title_guess": title,
            },
            "toc": build_toc(self.root),
            "content": self.root,
        }

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _open_section(
        self,
        title: str,
        level: int,
        page: int | None,
    ) -> dict:
        """Close deeper or equal sections, then open a new one."""
        while len(self.stack) > 1 and self.stack[-1]["level"] >= level:
            self.stack.pop()

        section = new_section(title, level, page)
        self.stack[-1]["subsections"].append(section)
        self.stack.append(section)

        return section

    def _add_heading(self, text: str, page: int | None) -> None:
        level = detect_logical_level(text)
        section = self._open_section(text, level, page)

        # The paper title is the first heading, provided it is neither
        # numbered (e.g. "I. INTRODUCTION") nor an "Abstract" heading.
        is_first_heading = len(self.root["subsections"]) == 1
        is_title = (
            is_first_heading
            and not has_section_number(text)
            and split_abstract(text) is None
        )

        if is_title:
            self.title_section = section
        else:
            self.in_front_matter = False

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _add_block(self, block: dict) -> None:
        if self.in_front_matter and block["type"] == "text":
            abstract_body = split_abstract(block["text"])

            if abstract_body is not None:
                self._add_abstract(block, abstract_body)
                return

            # Text directly under the paper title: author information.
            if self.stack[-1] is self.title_section:
                self._open_section(AUTHORS_TITLE, 1, block["page"])

        self._append(block)

    def _add_abstract(self, block: dict, body: str) -> None:
        self.in_front_matter = False
        self._open_section(ABSTRACT_TITLE, 1, block["page"])

        # Skip empty blocks (the paragraph contained only the word "Abstract").
        if body:
            block["text"] = body
            self._append(block)

    def _append(self, block: dict) -> None:
        """Attach section context to the block and store it."""
        context = get_section_context(self.stack)

        if context["section"] is not None:
            block["section"] = context["section"]

        if context["subsection"] is not None:
            block["subsection"] = context["subsection"]

        block["section_path"] = context["section_path"]

        self.stack[-1]["blocks"].append(block)


def build_document(
    items: list,
    base_dir: Path,
    source: str,
) -> dict:
    builder = _TreeBuilder(base_dir)

    for item in items:
        builder.add(item)

    return builder.to_document(source)


# ----------------------------------------------------------------------
# Tree utilities
# ----------------------------------------------------------------------


def iter_tables(section: dict):
    """Yield every table block in the section and its subsections."""
    for block in section.get("blocks", []):
        if block.get("type") == "table":
            yield block

    for subsection in section.get("subsections", []):
        yield from iter_tables(subsection)


def count_blocks(
    section: dict,
    counts: dict | None = None,
) -> dict:
    """Count blocks by type across the whole section tree."""
    if counts is None:
        counts = {}

    for block in section.get("blocks", []):
        block_type = block.get("type", "unknown")
        counts[block_type] = counts.get(block_type, 0) + 1

    for subsection in section.get("subsections", []):
        count_blocks(subsection, counts)

    return counts