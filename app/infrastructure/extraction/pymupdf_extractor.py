"""
Primary extractor: PyMuPDF (fitz).

Recovers, in a single pass:
  - document metadata (title, authors, page count)
  - reading-ordered text blocks with page + bbox
  - images (raw bytes) with page + bbox
  - tables via pdfplumber (see pdfplumber_extractor.py for fallback)

Headings and section hierarchy are reconstructed by
`section_tree_builder.py`, and captions are attached by
`caption_matcher.py`.
"""
from __future__ import annotations

import hashlib
import logging
import os

import fitz  # PyMuPDF

from app.domain.entities.document import Document, DocumentMetadata
from app.domain.entities.element import DocumentElement, ElementType
from app.domain.entities.image import ExtractedImage
from app.domain.exceptions import ExtractionError
from app.domain.repositories.document_extractor import DocumentExtractorPort
from app.domain.value_objects.ids import DocumentId, ImageId
from app.infrastructure.extraction.caption_matcher import attach_captions
from app.infrastructure.extraction.pdfplumber_extractor import extract_tables_pdfplumber
from app.infrastructure.extraction.section_tree_builder import build_sections_and_elements

logger = logging.getLogger("paperlens.infrastructure.extraction.pymupdf")


class PyMuPDFExtractor(DocumentExtractorPort):
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def extract(self, file_path: str) -> Document:
        try:
            pdf = fitz.open(file_path)
        except Exception as exc:
            raise ExtractionError(f"Cannot open PDF '{file_path}': {exc}") from exc

        document_id = DocumentId(self._stable_id(file_path))
        warnings: list[str] = []

        meta = pdf.metadata or {}
        doc_meta = DocumentMetadata(
            document_id=document_id,
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path),
            page_count=pdf.page_count,
            title=(meta.get("title") or "").strip() or None,
            authors=[a.strip() for a in (meta.get("author") or "").split(";") if a.strip()],
        )

        elements: list[DocumentElement] = []
        images: list[ExtractedImage] = []
        order = 0

        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            page_number = page_index + 1

            # --- Text blocks in reading order ---
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception as exc:
                warnings.append(f"Page {page_number}: text block extraction failed ({exc})")
                blocks = []

            for block in blocks:
                if block.get("type") != 0:
                    continue
                block_text = self._block_text(block)
                if not block_text.strip():
                    continue
                element = DocumentElement(
                    element_id=f"{document_id}-p{page_number}-e{order}",
                    document_id=document_id,
                    element_type=ElementType.PARAGRAPH,
                    order=order,
                    text=block_text,
                    page_number=page_number,
                    bbox=tuple(block["bbox"]),
                    extra={"avg_font_size": self._avg_font_size(block)},
                )
                elements.append(element)
                order += 1

            # --- Images ---
            try:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base = pdf.extract_image(xref)
                    raw = base["image"]
                    image_id = ImageId(f"{document_id}_img_{page_number:03d}_{img_index:03d}")
                    bbox = self._image_bbox(page, xref)
                    images.append(ExtractedImage(
                        image_id=image_id,
                        document_id=document_id,
                        page_number=page_number,
                        section_path=None,  # type: ignore[arg-type]  # filled by section_tree_builder
                        image_path="",        # filled by ImageStorePort on save
                        bbox=bbox,
                        width=base.get("width"),
                        height=base.get("height"),
                        extra={"_raw_bytes": raw, "_ext": base.get("ext", "png")},
                    ))
            except Exception as exc:
                warnings.append(f"Page {page_number}: image extraction failed ({exc})")

        # --- Tables (via pdfplumber) ---
        tables = []
        try:
            tables = extract_tables_pdfplumber(file_path, document_id)
        except Exception as exc:
            warnings.append(f"Table extraction failed ({exc})")

        # --- Section hierarchy + element tagging ---
        elements = build_sections_and_elements(elements, doc_meta)
        # Attach section paths to images and tables by page proximity.
        self._tag_assets_with_sections(elements, images, tables)
        attach_captions(elements, images, tables)

        pdf.close()

        return Document(
            metadata=doc_meta,
            elements=elements,
            images=images,
            tables=tables,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stable_id(file_path: str) -> str:
        h = hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()
        return f"doc_{h[:12]}"

    @staticmethod
    def _block_text(block: dict) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            if line_text.strip():
                lines.append(line_text)
        return "\n".join(lines)

    @staticmethod
    def _avg_font_size(block: dict) -> float:
        sizes: list[float] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(float(span.get("size", 0)))
        return sum(sizes) / len(sizes) if sizes else 0.0

    @staticmethod
    def _image_bbox(page, xref: int):
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                return (r.x0, r.y0, r.x1, r.y1)
        except Exception:
            pass
        return None

    @staticmethod
    def _tag_assets_with_sections(elements, images, tables) -> None:
        """Assign each image/table the section path of the closest
        preceding text element on the same page."""
        from app.domain.value_objects.section_path import SectionPath

        by_page: dict[int, list] = {}
        for e in elements:
            by_page.setdefault(e.page_number or 0, []).append(e)

        def closest_section(page_number: int):
            candidates = by_page.get(page_number) or by_page.get(page_number - 1) or []
            return candidates[-1].section_path if candidates else SectionPath.empty()

        for img in images:
            img.section_path = closest_section(img.page_number)
        for tbl in tables:
            tbl.section_path = closest_section(tbl.page_number)