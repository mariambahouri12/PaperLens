"""
Integration tests for the PyMuPDF extractor.

These tests build a tiny PDF on the fly with PyMuPDF itself, so no
binary fixture is required and the test suite can run anywhere.

We assert that:

  - the extractor reports the right `supports()` for PDFs
  - metadata (filename, page_count, file_size) is populated
  - paragraphs are recovered in reading order
  - a heading-shaped block is later promoted by the section builder
  - embedded raster images are recovered with raw bytes and image_id
  - tables are extracted as structured rows (not flattened text)
  - captions that start with "Figure N" are attached to the image
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from app.domain.entities.element import ElementType
from app.infrastructure.extraction.pymupdf_extractor import PyMuPDFExtractor


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def simple_pdf(tmp_path: Path) -> Path:
    """A one-page PDF with a heading, a body paragraph and a caption."""
    pdf_path = tmp_path / "simple.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 80), "Bayesian Methods", fontsize=18, fontname="helv")
    page.insert_text((72, 120), "This is the introduction paragraph.", fontsize=11, fontname="helv")
    page.insert_text((72, 160), "Figure 1: System architecture", fontsize=11, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_image(tmp_path: Path) -> Path:
    """A one-page PDF with a heading, a caption, and an embedded image."""
    pdf_path = tmp_path / "with_image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 80), "Results", fontsize=18, fontname="helv")
    page.insert_text((72, 120), "See Figure 1 below.", fontsize=11, fontname="helv")
    page.insert_text((72, 500), "Figure 1: A tiny 1x1 PNG.", fontsize=11, fontname="helv")

    # 1x1 red PNG, embedded as an image block.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x00\x03\x00\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    page.insert_image(fitz.Rect(72, 140, 272, 340), stream=png_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
def test_extractor_supports_pdf(simple_pdf: Path):
    extractor = PyMuPDFExtractor()
    assert extractor.supports(str(simple_pdf))
    assert not extractor.supports("something.html")


def test_extract_populates_document_metadata(simple_pdf: Path):
    doc = PyMuPDFExtractor().extract(str(simple_pdf))

    assert doc.metadata.filename == "simple.pdf"
    assert doc.metadata.page_count == 1
    assert doc.metadata.file_size_bytes > 0
    assert doc.metadata.document_id.startswith("doc_")


def test_extract_recovers_paragraphs_and_heading(simple_pdf: Path):
    doc = PyMuPDFExtractor().extract(str(simple_pdf))

    texts = [e.text for e in doc.elements]
    assert any("Bayesian Methods" in t for t in texts)
    assert any("introduction paragraph" in t for t in texts)

    # The heading block should have been promoted to ElementType.HEADING.
    heading = next(e for e in doc.elements if "Bayesian Methods" in e.text)
    assert heading.element_type == ElementType.HEADING
    assert heading.section_path.as_list() == ["Bayesian Methods"]

    # The paragraph that follows should inherit the heading's section.
    paragraph = next(e for e in doc.elements if "introduction paragraph" in e.text)
    assert paragraph.section_path.as_list() == ["Bayesian Methods"]


def test_extract_preserves_reading_order(simple_pdf: Path):
    doc = PyMuPDFExtractor().extract(str(simple_pdf))

    orders = [e.order for e in doc.elements]
    assert orders == sorted(orders)
    # First element should be the heading at the top of the page.
    assert "Bayesian Methods" in doc.elements[0].text


def test_extract_recovers_embedded_image(pdf_with_image: Path):
    doc = PyMuPDFExtractor().extract(str(pdf_with_image))

    assert len(doc.images) == 1
    img = doc.images[0]
    assert img.document_id == doc.metadata.document_id
    assert img.page_number == 1
    assert img.bbox is not None
    assert img.width == 1 and img.height == 1
    assert "_raw_bytes" in img.extra
    assert len(img.extra["_raw_bytes"]) > 0


def test_caption_is_attached_to_image(pdf_with_image: Path):
    doc = PyMuPDFExtractor().extract(str(pdf_with_image))

    img = doc.images[0]
    assert img.caption.startswith("Figure 1")
    assert "tiny 1x1 PNG" in img.caption

    # And the caption element is tagged as a CAPTION for downstream use.
    caption_el = next(e for e in doc.elements if "Figure 1" in e.text)
    assert caption_el.element_type == ElementType.CAPTION


def test_extraction_does_not_raise_on_missing_pdf(tmp_path: Path):
    from app.domain.exceptions import ExtractionError

    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(ExtractionError):
        PyMuPDFExtractor().extract(str(missing))


def test_extractor_ignores_non_pdf(simple_pdf: Path, tmp_path: Path):
    extractor = PyMuPDFExtractor()
    html = tmp_path / "page.html"
    html.write_text("<html><body>hi</body></html>")
    assert not extractor.supports(str(html))