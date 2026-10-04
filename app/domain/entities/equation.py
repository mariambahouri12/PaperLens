from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.domain.value_objects.ids import DocumentId
from app.domain.value_objects.section_path import SectionPath


@dataclass
class ExtractedEquation:
    """
    An equation extracted from a document.

    The equation is stored as LaTeX so that downstream layers can
    render it, index it, or include it in generated chunks.
    """

    equation_id: str
    document_id: DocumentId
    page_number: int
    section_path: SectionPath
    latex: str
    bbox: Optional[tuple[float, float, float, float]] = None
    extra: dict = field(default_factory=dict)