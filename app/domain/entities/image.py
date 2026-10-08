# domain/entities/image.py
"""
An image (or chart) extracted from a document.

The binary is NEVER loaded in memory by this entity: it lives on disk
at `image_path` (written by the extraction pipeline). The entity only
carries provenance and the file path.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.value_objects.ids import DocumentId, ImageId
from app.domain.value_objects.section_path import SectionPath


@dataclass(frozen=True)
class ExtractedImage:
    """
    Reference to an image on disk.

    `image_path` is the real filesystem path (absolute or relative to the
    extraction output). It is what `ImageResolverPort.resolve()` consumes.
    """

    image_id: ImageId
    document_id: DocumentId
    image_path: str
    page_number: int = 0
    section_path: SectionPath = SectionPath.empty()
    caption: str = ""
    footnote: str = ""
    kind: str = "image"  # "image" or "chart"