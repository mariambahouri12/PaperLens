from app.config.settings import Settings
from app.domain.entities.document import Document, DocumentMetadata
from app.domain.entities.element import DocumentElement, ElementType
from app.domain.value_objects.ids import DocumentId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.chunking.hierarchical_chunker import HierarchicalChunker


def _make_doc(elements):
    meta = DocumentMetadata(
        document_id=DocumentId("doc_test"),
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_size_bytes=123,
        page_count=1,
    )
    return Document(metadata=meta, elements=elements)


def test_small_section_becomes_single_chunk():
    settings = Settings(section_max_tokens=1000, chunk_size=500, chunk_overlap=50)
    chunker = HierarchicalChunker(settings)
    path = SectionPath.of(["Intro"])
    elements = [
        DocumentElement("e1", DocumentId("doc_test"), ElementType.PARAGRAPH, 0,
                        text="Short paragraph.", page_number=1, section_path=path),
    ]
    chunks = chunker.chunk(_make_doc(elements))
    assert len(chunks) == 1
    assert chunks[0].metadata.section_path.as_list() == ["Intro"]


def test_large_section_is_split_with_overlap():
    settings = Settings(section_max_tokens=30, chunk_size=20, chunk_overlap=5)
    chunker = HierarchicalChunker(settings)
    path = SectionPath.of(["Big"])
    paragraphs = [
        DocumentElement(f"e{i}", DocumentId("doc_test"), ElementType.PARAGRAPH, i,
                        text=" ".join(["word"] * 20), page_number=1, section_path=path)
        for i in range(10)
    ]
    chunks = chunker.chunk(_make_doc(paragraphs))
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.section_path.as_list() == ["Big"]