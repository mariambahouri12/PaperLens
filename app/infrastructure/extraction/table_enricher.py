# extraction/table_enricher.py
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from . import api
from .document_builder import iter_tables
from .table_parser import html_to_rows


GEMINI_HINT = (
    "Extract only the table grid (header and data rows). Ignore the caption, title, "
    "footnotes and any text outside the grid. Every value must be a JSON string exactly "
    'as printed (for example "5.00" or "100K"), never a JSON number.'
)


def apply_table(
    block: dict,
    columns: list,
    rows: list,
    source: str,
    warnings: list | None = None,
) -> None:
    cols = api.unique(columns)

    clean = []
    bad = 0

    for row in rows:
        row = list(row)

        if len(row) != len(cols):
            bad += 1

            row = (
                row
                + [None] * len(cols)
            )[:len(cols)]

        clean.append([
            None if value is None
            else str(value)
            for value in row
        ])

    block.pop(
        "mineru_html",
        None,
    )

    block["source"] = source
    block["columns"] = cols
    block["rows"] = clean

    warning_list = list(
        warnings or []
    )

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
    rows = html_to_rows(
        block.get(
            "mineru_html",
            "",
        )
    )

    warnings = [
        f"Gemini failed ({error}); "
        f"MinerU version preserved"
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

    block.pop(
        "mineru_html",
        None,
    )

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

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = api.get_client()
    models = api.pick_models(client)

    start = time.perf_counter()

    for number, block in enumerate(
        tables,
        1,
    ):
        block["id"] = f"table_{number}"

        cache_file = (
            cache_dir
            / f"table_{number}.json"
        )

        label = (
            f"[{number}/{len(tables)}] "
            f"{block['id']} "
            f"(page {block.get('page')})"
        )

        try:
            if (
                cache_file.is_file()
                and not force
            ):
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

                image = Image.open(
                    image_path
                ).convert("RGB")

                raw = api.call_gemini(
                    client,
                    models,
                    image,
                    extra_hint=GEMINI_HINT,
                )

                found = api.parse_json(raw)

                if not found:
                    raise ValueError(
                        "No table detected by Gemini"
                    )

                if len(found) > 1:
                    print(
                        f"  {len(found)} tables detected "
                        f"in the image; keeping the longest one"
                    )

                table = max(
                    found,
                    key=lambda value: len(
                        value.get("rows", [])
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
                    api.PAUSE_BETWEEN_IMAGES
                )

            apply_table(
                block,
                table.get("columns", []),
                table.get("rows", []),
                "gemini",
            )

            stats["tables_ok"] += 1

        except (
            Exception,
            SystemExit,
        ) as error:
            print(
                f"  FAILED: {error}"
            )

            apply_fallback(
                block,
                str(error),
            )

            stats["tables_failed"] += 1

    stats["gemini_time"] = (
        time.perf_counter() - start
    )