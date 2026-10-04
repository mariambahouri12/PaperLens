from __future__ import annotations


def join_value(value) -> str:
    if isinstance(value, list):
        return " ".join(
            str(item) for item in value
        ).strip()

    return str(value or "").strip()


def format_size(value: float) -> str:
    value = float(value)

    for unit in ("B", "KB", "MB"):
        if value < 1024:
            if unit == "B":
                return f"{value:.0f} B"

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value / 1024:.2f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return (
            f"{minutes} min "
            f"{remaining_seconds:.2f} sec"
        )

    return (
        f"{minutes // 60} h "
        f"{minutes % 60:.2f} min"
    )