#extraction/extractor.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .document_builder import (
    build_document,
    count_blocks,
)
from .markdown_renderer import (
    document_to_markdown,
)
from .runner import (
    find_content_list,
    run_mineru,
)
from .statistics import (
    print_statistics,
    print_toc,
)
from .table_enricher import (
    enrich_tables,
)


class MinerUExtractor:
    """
    Main orchestration service for the MinerU extraction pipeline.

    The extraction flow is:

        PDF
          |
          v
        MinerU
          |
          v
        content_list.json
          |
          v
        document tree
          |
          v
        Gemini table enrichment
          |
          +---- Gemini success -> Gemini table
          |
          +---- Gemini failure -> MinerU fallback
          |
          v
        JSON + Markdown
    """

    def __init__(
        self,
        output_dir: Path,
        backend: str = "pipeline",
        skip_mineru_tables: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.backend = backend
        self.skip_mineru_tables = (
            skip_mineru_tables
        )

    def extract(
        self,
        pdf_path: Path,
        no_run: bool = False,
        force_gemini: bool = False,
    ) -> dict:
        total_start = time.perf_counter()

        pdf_path = pdf_path.resolve()

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        raw_dir = (
            self.output_dir
            / "mineru_raw"
            / pdf_path.stem
        )

        cache_dir = (
            self.output_dir
            / "gemini_tables"
            / pdf_path.stem
        )

        stats = {
            "mineru_time": 0.0,
            "cpu_average": 0.0,
            "cpu_max": 0.0,
            "ram_average": 0,
            "ram_max": 0,
        }

        print("\n" + "=" * 70)
        print("PDF EXTRACTION")
        print("=" * 70)

        print(f"PDF                 : {pdf_path}")

        # ----------------------------------------------------------
        # MinerU
        # ----------------------------------------------------------

        if not no_run:
            run_mineru(
                pdf_path,
                raw_dir,
                self.backend,
                stats,
                self.skip_mineru_tables,
            )
        else:
            print(
                "\n--no-run mode: "
                "using existing MinerU output."
            )

        # ----------------------------------------------------------
        # Content list
        # ----------------------------------------------------------

        content_list = find_content_list(
            raw_dir
        )

        print(
            "\nReading:",
            content_list,
        )

        items = json.loads(
            content_list.read_text(
                encoding="utf-8"
            )
        )

        # ----------------------------------------------------------
        # Document tree
        # ----------------------------------------------------------

        document = build_document(
            items,
            content_list.parent,
            pdf_path.name,
        )

        print_toc(
            document["toc"]
        )

        # ----------------------------------------------------------
        # Gemini tables
        # ----------------------------------------------------------

        enrich_tables(
            document,
            cache_dir,
            force_gemini,
            stats,
        )

        # ----------------------------------------------------------
        # Output directory
        # ----------------------------------------------------------

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # JSON
        # ----------------------------------------------------------

        json_file = (
            self.output_dir
            / f"{pdf_path.stem}.json"
        )

        json_file.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"\nJSON written        : {json_file}"
        )

        # ----------------------------------------------------------
        # Markdown
        # ----------------------------------------------------------

        markdown_file = (
            self.output_dir
            / f"{pdf_path.stem}.md"
        )

        markdown_file.write_text(
            document_to_markdown(document),
            encoding="utf-8",
        )

        print(
            f"Markdown written    : "
            f"{markdown_file}"
        )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        print_statistics(
            pdf_path,
            stats,
            count_blocks(
                document["content"]
            ),
            time.perf_counter()
            - total_start,
        )

        return document


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
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory.",
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
        help=(
            "Reuse an existing MinerU output."
        ),
    )

    parser.add_argument(
        "--force-gemini",
        action="store_true",
        help=(
            "Ignore the Gemini table cache."
        ),
    )

    parser.add_argument(
        "--skip-mineru-tables",
        action="store_true",
        help=(
            "Disable table recognition in MinerU."
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    extractor = MinerUExtractor(
        output_dir=args.output,
        backend=args.backend,
        skip_mineru_tables=(
            args.skip_mineru_tables
        ),
    )

    extractor.extract(
        pdf_path=args.pdf,
        no_run=args.no_run,
        force_gemini=args.force_gemini,
    )


if __name__ == "__main__":
    main()