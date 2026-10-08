# app/infrastructure/extraction/table_enricher.py
"""
Table enrichment orchestration.

Flow:

    MinerU table
         |
         v
    Gemini
      /   \
 success  failure
   |        |
   v        v
 Gemini   MinerU fallback

Gemini is optional. If it is unavailable, the document remains usable
with the MinerU table representation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image

from app.domain.exceptions import GeminiError

from . import api
from .document_builder import iter_tables
from .gemini_client import (
    PAUSE_BETWEEN_IMAGES,
    get_client,
    pick_models,
)
from .gemini_prompts import GEMINI_TABLE_HINT
from .table_parser import html_to_rows

logger = logging.getLogger(
    "paperlens.infrastructure.table_enricher"
)


_EXPECTED_TABLE_ERRORS = (
    GeminiError,
    FileNotFoundError,
    ValueError,
    OSError,
)


def apply_table(
    block: dict,
    columns: list,
    rows: list,
    source: str,
    warnings: list | None = None,
) -> None:
    """
    Write normalized table data onto a table block.
    """
    cols = api.unique(columns)

    clean: list[list[str | None]] = []
    bad = 0

    for row in rows:
        row = list(row)

        if len(row) != len(cols):
            bad += 1
            row = (
                row + [None] * len(cols)
            )[:len(cols)]

        clean.append(
            [
                None if value is None else str(value)
                for value in row
            ]
        )

    block.pop("mineru_html", None)
    block["source"] = source
    block["columns"] = cols
    block["rows"] = clean

    warning_list = list(warnings or [])

    if bad:
        warning_list.append(
            f"{bad} row(s) had a number of values "
            f"different from the number of columns "
            f"(padded/truncated; please verify)"
        )

    if warning_list:
        block["warnings"] = warning_list


def apply_fallback(
    block: dict,
    error: str,
) -> None:
    """
    Populate a table block from MinerU HTML.
    """
    rows = html_to_rows(
        block.get("mineru_html", "")
    )

    warnings = [
        f"Gemini failed ({error}); "
        "MinerU version preserved"
    ]

    if rows:
        apply_table(
            block,
            rows[0],
            rows[1:],
            "mineru_fallback",
            warnings,
        )
        return

    block.pop("mineru_html", None)

    block.update(
        source="mineru_fallback",
        columns=[],
        rows=[],
        warnings=warnings,
    )


def enrich_tables(
    document: dict,
    cache_dir: Path,
    force: bool,
    stats: dict,
) -> None:
    """
    Enrich all document tables with Gemini when available.

    Every table-level Gemini failure falls back to MinerU.
    Unexpected programming errors are not swallowed.
    """
    tables = list(
        iter_tables(document["content"])
    )

    stats.update(
        tables_total=len(tables),
        tables_ok=0,
        tables_failed=0,
        tables_cached=0,
        gemini_time=0.0,
    )

    if not tables:
        print(
            "\nNo tables found in this document. "
            "Gemini will not be called."
        )
        return

    print("\n" + "=" * 70)
    print(
        f"GEMINI: {len(tables)} table(s) to process"
    )
    print("=" * 70)

    try:
        client = get_client()
        models = pick_models(client)

    except GeminiError as exc:
        print(
            f"Gemini unavailable ({exc}); "
            "using MinerU fallback for all tables."
        )

        logger.warning(
            "Gemini unavailable: %s",
            exc,
        )

        for number, block in enumerate(tables, 1):
            block["id"] = f"table_{number}"

            apply_fallback(
                block,
                "Gemini unavailable",
            )

            stats["tables_failed"] += 1

        return

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    for number, block in enumerate(tables, 1):
        block["id"] = f"table_{number}"

        cache_file = (
            cache_dir /
            f"table_{number}.json"
        )

        label = (
            f"[{number}/{len(tables)}] "
            f"{block['id']} "
            f"(page {block.get('page')})"
        )

        try:
            if cache_file.is_file() and not force:
                table = json.loads(
                    cache_file.read_text(
                        encoding="utf-8"
                    )
                )

                stats["tables_cached"] += 1
                print(f"{label}: cache")

            else:
                image_path = block.get(
                    "image_path"
                )

                if (
                    not image_path
                    or not Path(image_path).is_file()
                ):
                    raise FileNotFoundError(
                        "Table image not found"
                    )

                print(
                    f"{label}: sending to Gemini"
                )

                with Image.open(image_path) as source:
                    image = source.convert("RGB")

                    raw = api.call_gemini(
                        client,
                        models,
                        image,
                        extra_hint=GEMINI_TABLE_HINT,
                    )

                found = api.parse_json(raw)

                if not found:
                    raise ValueError(
                        "No table detected by Gemini"
                    )

                table = max(
                    found,
                    key=lambda value: len(
                        value["rows"]
                    ),
                )

                cache_file.write_text(
                    json.dumps(
                        table,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                time.sleep(
                    PAUSE_BETWEEN_IMAGES
                )

            apply_table(
                block,
                table["columns"],
                table["rows"],
                "gemini",
            )

            stats["tables_ok"] += 1

        except _EXPECTED_TABLE_ERRORS as error:
            print(
                f"  FAILED: {error}"
            )

            logger.warning(
                "Table %s failed: %s",
                block.get("id"),
                error,
            )

            apply_fallback(
                block,
                str(error),
            )

            stats["tables_failed"] += 1

    stats["gemini_time"] = (
        time.perf_counter() - start
    )