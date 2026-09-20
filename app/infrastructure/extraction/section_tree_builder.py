"""
Reconstruct the document's section hierarchy from heading-shaped
paragraphs (blocks with above-average font size on short lines), then
assign each element its full `SectionPath`.

Heuristic, but works well for typical research papers where headings
are visually distinct from body text.
"""
from __future__ import annotations

import statistics

from app.domain.entities.document import DocumentMetadata
from app.domain.entities.element import DocumentElement, ElementType
from app.domain.entities.section import Section
from app.domain.value_objects.section_path import SectionPath

_HEADING_MAX_CHARS = 120
_HEADING_SIZE_FACTOR = 1.15


def build_sections_and_elements(
    elements: list[DocumentElement],
    doc_meta: DocumentMetadata,
) -> list[DocumentElement]:
    if not elements:
        return elements

    # Compute body font size from the longest block (most likely body text).
    sizes = [e.extra.get("avg_font_size", 0.0) for e in elements if e.text]
    body_size = statistics.median(sizes) if sizes else 0.0

    # Build section tree.
    root = Section(title="", path=SectionPath.empty(), level=0)
    current = root
    stack: list[Section] = [root]

    for e in elements:
        is_heading = (
            e.extra.get("avg_font_size", 0.0) >= body_size * _HEADING_SIZE_FACTOR
            and len(e.text) <= _HEADING_MAX_CHARS
            and "\n" not in e.text.strip()
        )

        if is_heading:
            e.element_type = ElementType.HEADING
            level = 1 if not stack[-1].path else len(stack[-1].path.parts) + 1

            new_path = SectionPath.of([p for p in current.path.parts] + [e.text.strip()])

            # Pop up to the right parent level
            while len(stack) > 1 and len(stack[-1].path.parts) >= level:
                stack.pop()

            parent = stack[-1]
            section = Section(title=e.text.strip(), path=new_path, level=level)
            parent.add_child(section)
            stack.append(section)
            current = section

            e.section_path = new_path
            current.add_element(e)
        else:
            e.section_path = current.path
            current.add_element(e)

    # Attach a synthetic root title from metadata if available.
    if doc_meta.title and not root.children:
        root.title = doc_meta.title

    return elements