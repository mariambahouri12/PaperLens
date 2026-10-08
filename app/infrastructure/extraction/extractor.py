# app/infrastructure/extraction/extractor.py
"""
MinerU extraction pipeline, exposed as a domain port.

Output layout:

    data/mineru_raw/<stem>/.../<stem>_content_list.json
    data/mineru_raw/<stem>/.../images/*.jpg
    data/debug/<stem>.json
    data/debug/gemini_tables/<stem>/table_N.json

The `data/output/` directory is owned by the ingestion use case.

The document_id is a UUID v4. It is generated once and persisted in
the extraction JSON. Re-ingesting the same PDF reuses the stored id.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.config.settings import Settings
from app.domain.entities.document import Document
from app.domain.exceptions import ExtractionError, MinerUError
from app.domain.repositories.document_extractor import DocumentExtractorPort
from app.domain.value_objects.ids import DocumentId, new_document_id

from .document_builder import build_document, count_blocks
from .document_mapper import DocumentMapper
from .runner import find_content_list, run_mineru
from .statistics import print_statistics, print_toc
from .table_enricher import enrich_tables

logger = logging.getLogger("paperlens.infrastructure.extraction")


class MinerUExtractor(DocumentExtractorPort):
    """
    Main orchestration service for the MinerU extraction pipeline.

    Flow:

        PDF
          |
          v
        MinerU
          |
          v
        document tree
          |
          v
        Gemini table enrichment
          |
          v
        extraction JSON
          |
          v
        domain Document
    """

    SUPPORTED_SUFFIXES = frozenset({".pdf"})

    def __init__(
        self,
        settings: Settings,
        backend: str = "pipeline",
        skip_mineru_tables: bool = False,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.skip_mineru_tables = skip_mineru_tables
        self._mapper = DocumentMapper()

    # ------------------------------------------------------------------
    # DocumentExtractorPort
    # ------------------------------------------------------------------

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_SUFFIXES

    def extract(self, file_path: str) -> Document:
        """
        Run the extraction pipeline and return a domain Document.

        Existing document IDs are reused when an extraction JSON already
        exists for the same PDF.
        """
        path = Path(file_path).resolve()

        if not path.is_file():
            raise ExtractionError(f"File not found: {path}")

        json_file = self.settings.debug_dir / f"{path.stem}.json"

        # Read the existing ID BEFORE rebuilding the document.
        existing_document_id = self._load_existing_document_id(json_file)

        try:
            data = self._run_pipeline(
                pdf_path=path,
                no_run=json_file.is_file(),
                force_gemini=False,
            )
        except MinerUError as exc:
            raise ExtractionError(
                f"MinerU extraction failed for {path}: {exc}"
            ) from exc
        except ExtractionError:
            raise
        except (OSError, ValueError) as exc:
            raise ExtractionError(
                f"Extraction failed for {path}: {exc}"
            ) from exc

        document_id = existing_document_id or new_document_id()

        if existing_document_id:
            logger.info(
                "Reusing existing document_id: %s",
                existing_document_id,
            )
        else:
            logger.info(
                "Assigned new document_id: %s",
                document_id,
            )

        data.setdefault("metadata", {})["document_id"] = str(document_id)

        self.settings.debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            json_file.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ExtractionError(
                f"Could not write extraction JSON for {path}: {exc}"
            ) from exc

        logger.info(
            "Extraction JSON written: %s",
            json_file,
        )

        try:
            return self._mapper.to_document(
                data,
                document_id=document_id,
                file_path=path,
            )
        except (ValueError, KeyError) as exc:
            raise ExtractionError(
                f"Could not map extraction result for {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Existing document ID
    # ------------------------------------------------------------------

    @staticmethod
    def _load_existing_document_id(
        json_file: Path,
    ) -> DocumentId | None:
        """
        Read a previously persisted document_id if available.

        Invalid or missing metadata does not prevent a new extraction
        from receiving a fresh document ID.
        """
        if not json_file.is_file():
            return None

        try:
            data = json.loads(
                json_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read previous extraction metadata from %s: %s",
                json_file,
                exc,
            )
            return None

        metadata = data.get("metadata") or {}
        stored = metadata.get("document_id")

        if not stored:
            return None

        try:
            return DocumentId(stored)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Ignoring invalid stored document_id in %s: %s",
                json_file,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Procedural pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        pdf_path: Path,
        no_run: bool = False,
        force_gemini: bool = False,
    ) -> dict:
        """
        Run the full extraction pipeline and return the raw JSON dict.

        This method does not write the final extraction JSON.
        """
        total_start = time.perf_counter()

        pdf_path = pdf_path.resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        raw_dir = (
            self.settings.mineru_raw_dir /
            pdf_path.stem
        )

        cache_dir = (
            self.settings.debug_dir /
            "gemini_tables" /
            pdf_path.stem
        )

        stats = {
            "mineru_time": 0.0,
            "cpu_average": 0.0,
            "cpu_max": 0.0,
            "ram_average": 0,
            "ram_max": 0,
        }

        logger.info(
            "PDF extraction: %s",
            pdf_path,
        )

        if not no_run:
            run_mineru(
                pdf_path,
                raw_dir,
                self.backend,
                stats,
                self.skip_mineru_tables,
            )
        else:
            logger.info(
                "--no-run mode: using existing MinerU output"
            )

        content_list = find_content_list(raw_dir)

        logger.info(
            "Reading content list: %s",
            content_list,
        )

        try:
            items = json.loads(
                content_list.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ExtractionError(
                f"Could not read MinerU content list "
                f"{content_list}: {exc}"
            ) from exc

        document = build_document(
            items,
            content_list.parent,
            pdf_path.name,
        )

        print_toc(document["toc"])

        enrich_tables(
            document,
            cache_dir,
            force_gemini,
            stats,
        )

        print_statistics(
            pdf_path,
            stats,
            count_blocks(document["content"]),
            time.perf_counter() - total_start,
        )

        return document

    # ------------------------------------------------------------------
    # CLI / helpers
    # ------------------------------------------------------------------

    def _resolve_document_id(
        self,
        data: dict,
    ) -> DocumentId:
        """
        Resolve an ID already present in the data, or create a new one.

        Kept for compatibility with existing callers.
        """
        meta = data.get("metadata") or {}
        stored = meta.get("document_id")

        if stored:
            return DocumentId(stored)

        return new_document_id()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PDF extraction using MinerU "
            "and Gemini table enrichment."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF file.",
    )

    parser.add_argument(
        "-b",
        "--backend",
        default="pipeline",
        help="MinerU backend.",
    )

    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Reuse existing MinerU output.",
    )

    parser.add_argument(
        "--force-gemini",
        action="store_true",
        help="Ignore the Gemini table cache.",
    )

    parser.add_argument(
        "--skip-mineru-tables",
        action="store_true",
        help="Disable table recognition in MinerU.",
    )

    return parser


def main() -> None:
    from app.config.settings import settings as default_settings
    from app.infrastructure.logging.structured_logger import (
        configure_logging,
    )

    configure_logging(
        level=default_settings.log_level
    )

    parser = build_argument_parser()
    args = parser.parse_args()

    extractor = MinerUExtractor(
        settings=default_settings,
        backend=args.backend,
        skip_mineru_tables=args.skip_mineru_tables,
    )

    try:
        extractor._run_pipeline(
            pdf_path=args.pdf,
            no_run=args.no_run,
            force_gemini=args.force_gemini,
        )
    except (MinerUError, ExtractionError) as exc:
        logger.error(
            "Extraction failed: %s",
            exc,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()