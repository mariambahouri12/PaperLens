"""
Table extraction with pdfplumber.

Used by PyMuPDFExtractor to recover structured tables (rows, columns,
caption kept separate). Never flattens tables into plain text.
"""
from __future__ import annotations

import logging

import pdfplumber

from app.domain.entities.table import ExtractedTable
from app.domain.value_objects.ids import DocumentId, TableId
from app.domain.value_objects.section_path import SectionPath

logger = logging.getLogger("paperlens.infrastructure.extraction.pdfplumber")


def extract_tables_pdfplumber(file_path: str, document_id: DocumentId) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                try:
                    found = page.find_tables()
                except Exception as exc:
                    logger.warning("Page %s: table detection failed (%s)", page_index, exc)
                    continue
                for t_index, table in enumerate(found):
                    rows = table.extract() or []
                    rows = [[(c or "").strip() for c in row] for row in rows]
                    if not rows:
                        continue
                    tables.append(ExtractedTable(
                        table_id=TableId(f"{document_id}_tbl_{page_index:03d}_{t_index:03d}"),
                        document_id=document_id,
                        page_number=page_index,
                        section_path=SectionPath.empty(),
                        rows=rows,
                        bbox=tuple(table.bbox) if table.bbox else None,
                        has_header=True,
                    ))
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)
    return tables