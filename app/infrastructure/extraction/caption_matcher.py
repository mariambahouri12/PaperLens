"""
Attach captions to images and tables.

Heuristic rules:
  - A paragraph that starts with "Figure N" / "Fig. N" and is
    positioned near an image on the same page is treated as that
    image's caption.
  - A paragraph that starts with "Table N" and is near a table on the
    same page is treated as that table's caption.

Captions are also injected into the source element's text so they
remain visible in the extracted textual context.
"""
from __future__ import annotations

import re

from app.domain.entities.element import DocumentElement, ElementType
from app.domain.entities.image import ExtractedImage
from app.domain.entities.table import ExtractedTable

_FIG_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)
_TBL_RE = re.compile(r"^\s*table\s*\d+", re.IGNORECASE)


def attach_captions(
    elements: list[DocumentElement],
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
) -> None:
    # Index elements by page for quick lookup.
    by_page: dict[int, list[DocumentElement]] = {}
    for e in elements:
        by_page.setdefault(e.page_number or 0, []).append(e)

    # --- Figures ---
    for img in images:
        candidates = by_page.get(img.page_number, [])
        caption = _find_caption(candidates, _FIG_RE, img.bbox)
        if caption is not None:
            img.caption = caption.text.strip()
            caption.element_type = ElementType.CAPTION
            # Ensure the caption text stays part of the extracted content.
            if img.caption not in caption.text:
                caption.text = img.caption

    # --- Tables ---
    for tbl in tables:
        candidates = by_page.get(tbl.page_number, [])
        caption = _find_caption(candidates, _TBL_RE, tbl.bbox)
        if caption is not None:
            tbl.caption = caption.text.strip()
            caption.element_type = ElementType.CAPTION
            if tbl.caption not in caption.text:
                caption.text = tbl.caption


def _find_caption(
    candidates: list[DocumentElement],
    pattern: re.Pattern,
    bbox,
) -> DocumentElement | None:
    matches = [e for e in candidates if pattern.match(e.text)]
    if not matches:
        return None
    if bbox is None or len(matches) == 1:
        return matches[0]

    def distance(e: DocumentElement) -> float:
        if not e.bbox:
            return 1e9
        ex0, ey0, ex1, ey1 = e.bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        ecx = (ex0 + ex1) / 2
        ecy = (ey0 + ey1) / 2
        return ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5

    return min(matches, key=distance)