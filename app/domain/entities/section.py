from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.entities.element import DocumentElement
from app.domain.value_objects.section_path import SectionPath


@dataclass
class Section:
    """
    A node in the document's section hierarchy.

    A section owns its title, its direct elements (paragraphs, figures,
    tables ...) and its children sections. The chunker walks this tree
    to build retrieval units that preserve the full section path.
    """

    title: str
    path: SectionPath
    level: int = 1
    elements: list[DocumentElement] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    def add_element(self, element: DocumentElement) -> None:
        self.elements.append(element)

    def add_child(self, child: "Section") -> None:
        self.children.append(child)

    def iter_elements_depth_first(self):
        for element in self.elements:
            yield element
        for child in self.children:
            yield from child.iter_elements_depth_first()