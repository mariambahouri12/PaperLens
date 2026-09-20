from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.domain.value_objects.ids import DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath


class ElementType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    EQUATION = "equation"
    OTHER = "other"


@dataclass
class DocumentElement:
    """
    One atomic element in the document, in reading order.

    `text` is the textual content for text-like elements, and is left
    empty for FIGURE/TABLE references (their payload lives in the
    associated ExtractedImage/ExtractedTable). `image_id` and `table_id`
    are set for FIGURE and TABLE references respectively, so downstream
    layers can resolve the actual asset without embedding it in text.
    """

    element_id: str
    document_id: DocumentId
    element_type: ElementType
    order: int
    text: str = ""
    page_number: Optional[int] = None
    section_path: SectionPath = field(default_factory=SectionPath.empty)
    bbox: Optional[tuple[float, float, float, float]] = None
    image_id: Optional[ImageId] = None
    table_id: Optional[TableId] = None
    extra: dict = field(default_factory=dict)