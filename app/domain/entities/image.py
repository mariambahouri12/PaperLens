from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.domain.value_objects.ids import DocumentId, ImageId
from app.domain.value_objects.section_path import SectionPath


@dataclass
class ExtractedImage:
    """
    An image extracted from a document, stored on disk, referenced by
    `image_id` everywhere else in the pipeline.
    """

    image_id: ImageId
    document_id: DocumentId
    page_number: int
    section_path: SectionPath
    image_path: str
    caption: str = ""
    bbox: Optional[tuple[float, float, float, float]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extra: dict = field(default_factory=dict)