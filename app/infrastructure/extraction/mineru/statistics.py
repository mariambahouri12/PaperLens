from __future__ import annotations

from pathlib import Path

from .utils import format_duration, format_size


def print_statistics(
    pdf: Path,
    stats: dict,
    block_counts: dict,
    total_time: float,
) -> None:
    print("\n" + "=" * 70)
    print("PROCESSING STATISTICS")
    print("=" * 70)

    print(
        f"PDF size            : "
        f"{format_size(pdf.stat().st_size)}"
    )

    print(
        f"MinerU time         : "
        f"{format_duration(stats.get('mineru_time', 0))}"
    )

    print(
        f"Gemini time         : "
        f"{format_duration(stats.get('gemini_time', 0))}"
    )

    print(
        f"Total time          : "
        f"{format_duration(total_time)}"
    )

    print(
        f"MinerU average CPU  : "
        f"{stats.get('cpu_average', 0):.2f}%"
    )

    print(
        f"MinerU maximum CPU  : "
        f"{stats.get('cpu_max', 0):.2f}%"
    )

    print(
        f"MinerU average RAM  : "
        f"{format_size(stats.get('ram_average', 0))}"
    )

    print(
        f"MinerU maximum RAM  : "
        f"{format_size(stats.get('ram_max', 0))}"
    )

    print(
        f"\nTables              : "
        f"{stats.get('tables_total', 0)} "
        f"(Gemini OK: "
        f"{stats.get('tables_ok', 0)}, "
        f"cached: "
        f"{stats.get('tables_cached', 0)}, "
        f"MinerU fallback: "
        f"{stats.get('tables_failed', 0)})"
    )

    print("\nExtracted blocks:")

    for block_type, count in sorted(
        block_counts.items()
    ):
        print(
            f"  - {block_type:<12}: {count}"
        )

    print("=" * 70)


def print_toc(toc: list) -> None:
    print("\n" + "=" * 70)
    print("DETECTED TABLE OF CONTENTS")
    print("=" * 70)

    for level, title, page in toc:
        indent = "    " * max(
            level - 1,
            0,
        )

        print(
            f"{indent}- {title}"
            f"  [level={level}, page={page}]"
        )

    print("=" * 70)