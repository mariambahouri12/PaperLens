"""
Unit tests for the section hierarchy reconstruction.

We build synthetic documents from `DocumentElement` lists (no real PDF
needed) and assert that:

  - headings are promoted to ElementType.HEADING
  - each element gets the *full* section path (not just the leaf)
  - sibling and nested sections produce the expected tree
  - body paragraphs that happen to be long are not mistaken for headings
  - elements before the first heading inherit an empty (root) section path
"""
from __future__ import annotations

from app.domain.entities.document import DocumentMetadata
from app.domain.entities.element import DocumentElement, ElementType
from app.domain.value_objects.ids import DocumentId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.extraction.section_tree_builder import (
    build_sections_and_elements,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _meta() -> DocumentMetadata:
    return DocumentMetadata(
        document_id=DocumentId("doc_test"),
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_size_bytes=1,
        page_count=1,
    )


def _element(
    order: int,
    text: str,
    *,
    font_size: float = 10.0,
    page: int = 1,
) -> DocumentElement:
    return DocumentElement(
        element_id=f"e{order}",
        document_id=DocumentId("doc_test"),
        element_type=ElementType.PARAGRAPH,
        order=order,
        text=text,
        page_number=page,
        section_path=SectionPath.empty(),
        extra={"avg_font_size": font_size},
    )


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
def test_heading_is_promoted_and_body_keeps_leaf_path():
    # Body font size is 10; headings are set to 14 to clearly exceed the factor.
    elements = [
        _element(0, "Introduction", font_size=14.0),
        _element(1, "This is the introduction paragraph.", font_size=10.0),
    ]

    result = build_sections_and_elements(elements, _meta())

    assert result[0].element_type == ElementType.HEADING
    assert result[0].section_path.as_list() == ["Introduction"]

    assert result[1].element_type == ElementType.PARAGRAPH
    assert result[1].section_path.as_list() == ["Introduction"]


def test_nested_sections_build_full_path():
    elements = [
        _element(0, "Bayesian Methods", font_size=16.0),
        _element(1, "Methodology body.", font_size=10.0),
        _element(2, "Results", font_size=14.0),
        _element(3, "Experimental Results", font_size=13.0),
        _element(4, "We observe that ...", font_size=10.0),
        _element(5, "Ablation Study", font_size=13.0),
        _element(6, "Ablation body.", font_size=10.0),
        _element(7, "Conclusion", font_size=14.0),
        _element(8, "Concluding remarks.", font_size=10.0),
    ]

    result = build_sections_and_elements(elements, _meta())

    paths = {e.text: e.section_path.as_list() for e in result}

    assert paths["Bayesian Methods"] == ["Bayesian Methods"]
    assert paths["Methodology body."] == ["Bayesian Methods"]
    assert paths["Results"] == ["Bayesian Methods", "Results"]
    assert paths["Experimental Results"] == ["Bayesian Methods", "Results", "Experimental Results"]
    assert paths["We observe that ..."] == [
        "Bayesian Methods",
        "Results",
        "Experimental Results",
    ]
    assert paths["Ablation Study"] == ["Bayesian Methods", "Results", "Ablation Study"]
    assert paths["Ablation body."] == ["Bayesian Methods", "Results", "Ablation Study"]
    assert paths["Conclusion"] == ["Bayesian Methods", "Conclusion"]
    assert paths["Concluding remarks."] == ["Bayesian Methods", "Conclusion"]


def test_long_paragraph_is_not_promoted_to_heading_even_with_bigger_font():
    long_text = "word " * 60  # well beyond the 120-char heuristic
    elements = [
        _element(0, "Intro", font_size=14.0),
        _element(1, long_text, font_size=15.0),  # big font but too long
    ]

    result = build_sections_and_elements(elements, _meta())

    assert result[1].element_type == ElementType.PARAGRAPH
    assert result[1].section_path.as_list() == ["Intro"]


def test_multiline_block_is_not_promoted_to_heading():
    elements = [
        _element(0, "Intro", font_size=14.0),
        _element(1, "Line one\nLine two", font_size=14.0),  # contains a newline
    ]

    result = build_sections_and_elements(elements, _meta())

    assert result[1].element_type == ElementType.PARAGRAPH
    assert result[1].section_path.as_list() == ["Intro"]


def test_elements_before_first_heading_have_empty_section_path():
    elements = [
        _element(0, "Some preamble abstract text.", font_size=10.0),
        _element(1, "Another preamble line.", font_size=10.0),
    ]

    result = build_sections_and_elements(elements, _meta())

    for e in result:
        assert e.section_path.as_list() == []


def test_empty_input_is_returned_unchanged():
    assert build_sections_and_elements([], _meta()) == []


def test_heading_section_path_is_its_own_path():
    """A heading's section_path must equal its own title path, so that
    downstream layers can attach assets that live directly under the
    heading."""
    elements = [
        _element(0, "Top", font_size=16.0),
        _element(1, "Child", font_size=14.0),
    ]

    result = build_sections_and_elements(elements, _meta())

    assert result[0].section_path.as_list() == ["Top"]
    assert result[1].section_path.as_list() == ["Top", "Child"]