#extraction/statistics.py
"""Console reporting for the extraction pipeline (TOC + run statistics)."""
from __future__ import annotations

from pathlib import Path

LINE = "=" * 70


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds or 0), 0.0)

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, rest = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {rest:04.1f}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h {int(minutes):02d}m {rest:04.1f}s"


def format_bytes(value) -> str:
    size = float(value or 0)

    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} TB"


# ----------------------------------------------------------------------
# Table of contents
# ----------------------------------------------------------------------


def print_toc(toc: list) -> None:
    """Print the section tree. `toc` is a list of [level, title, page]."""
    print("\n" + LINE)
    print("TABLE OF CONTENTS")
    print(LINE)

    if not toc:
        print("No headings detected.")
        return

    for level, title, page in toc:
        indent = "  " * max(int(level) - 1, 0)
        location = f"  (p. {page})" if page else ""

        print(f"{indent}{title}{location}")


# ----------------------------------------------------------------------
# Run statistics
# ----------------------------------------------------------------------


def print_statistics(
    pdf_path: Path,
    stats: dict,
    block_counts: dict,
    total_time: float,
) -> None:
    pdf_path = Path(pdf_path)

    print("\n" + LINE)
    print("STATISTICS")
    print(LINE)

    size = pdf_path.stat().st_size if pdf_path.is_file() else 0
    print(f"PDF                 : {pdf_path.name} ({format_bytes(size)})")

    # MinerU ------------------------------------------------------------
    print("\nMinerU")
    print(f"  Time              : {format_duration(stats.get('mineru_time', 0))}")
    print(
        f"  CPU               : avg {stats.get('cpu_average', 0):.0f}% "
        f"| max {stats.get('cpu_max', 0):.0f}%"
    )
    print(
        f"  RAM               : avg {format_bytes(stats.get('ram_average', 0))} "
        f"| max {format_bytes(stats.get('ram_max', 0))}"
    )

    # Blocks ------------------------------------------------------------
    print("\nBlocks")

    if block_counts:
        for block_type, number in sorted(block_counts.items()):
            print(f"  {block_type:<18}: {number}")

        print(f"  {'total':<18}: {sum(block_counts.values())}")
    else:
        print("  none")

    # Gemini ------------------------------------------------------------
    total_tables = stats.get("tables_total", 0)

    print("\nGemini tables")

    if total_tables:
        print(f"  Tables            : {total_tables}")
        print(f"  Succeeded         : {stats.get('tables_ok', 0)}")
        print(f"  From cache        : {stats.get('tables_cached', 0)}")
        print(f"  Failed (fallback) : {stats.get('tables_failed', 0)}")
        print(f"  Time              : {format_duration(stats.get('gemini_time', 0))}")
    else:
        print("  No tables")

    print(f"\nTotal time          : {format_duration(total_time)}")
    print(LINE)