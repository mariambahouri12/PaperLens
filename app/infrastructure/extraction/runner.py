# extraction/runner.py
"""
MinerU subprocess runner and output finder.

Responsibilities (and ONLY these):
  - locate the `mineru` executable,
  - run it with the right arguments,
  - monitor CPU / RAM while it runs,
  - collect statistics,
  - raise `MinerUError` on failure.

The runner NEVER calls `sys.exit()`: callers receive an exception and
decide what to do (skip document, fall back, abort, ...).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from app.domain.exceptions import MinerUError


def monitor_process(
    process: subprocess.Popen,
    stats: dict,
    stop_event: threading.Event,
) -> None:
    """
    Sample CPU and RAM usage of the MinerU process (and its children)
    until `stop_event` is set, then store averages and maxima in `stats`.
    """
    cpu_values: list[float] = []
    ram_values: list[int] = []

    time.sleep(0.2)

    try:
        parent = psutil.Process(process.pid)
        parent.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        return

    while not stop_event.is_set():
        try:
            processes = [parent]
            try:
                processes.extend(parent.children(recursive=True))
            except psutil.NoSuchProcess:
                pass

            cpu = 0.0
            ram = 0

            for proc in processes:
                try:
                    cpu += proc.cpu_percent(interval=None)
                    ram += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            cpu_values.append(cpu)
            ram_values.append(ram)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        stop_event.wait(0.5)

    if cpu_values:
        stats["cpu_average"] = sum(cpu_values) / len(cpu_values)
        stats["cpu_max"] = max(cpu_values)

    if ram_values:
        stats["ram_average"] = sum(ram_values) / len(ram_values)
        stats["ram_max"] = max(ram_values)


def run_mineru(
    pdf: Path,
    raw_dir: Path,
    backend: str,
    stats: dict,
    skip_tables: bool,
) -> None:
    """
    Run MinerU on `pdf`, writing its raw output to `raw_dir`.

    Raises
    ------
    MinerUError
        If the `mineru` executable is missing or MinerU exits with a
        non-zero return code.
    """
    executable = shutil.which("mineru")

    if executable is None:
        raise MinerUError(
            "Command 'mineru' was not found. "
            "Activate the MinerU virtual environment."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "-p", str(pdf),
        "-o", str(raw_dir),
        "-b", backend,
    ]

    if skip_tables:
        command.extend(["-t", "false"])

    print("\n" + "=" * 70)
    print("MINERU")
    print("=" * 70)
    print("Command:", " ".join(command))
    print()

    start = time.perf_counter()

    process = subprocess.Popen(command)

    monitor_stats: dict = {}
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_process,
        args=(process, monitor_stats, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    return_code = process.wait()

    stop_event.set()
    monitor_thread.join()

    stats["mineru_time"] = time.perf_counter() - start

    defaults = {
        "cpu_average": 0.0,
        "cpu_max": 0.0,
        "ram_average": 0,
        "ram_max": 0,
    }
    for key, default in defaults.items():
        stats[key] = monitor_stats.get(key, default)

    if return_code != 0:
        raise MinerUError(
            f"MinerU failed with return code {return_code}."
        )

    print("\nMinerU completed successfully.")


def find_content_list(raw_dir: Path) -> Path:
    """
    Find the most relevant `*content_list*.json` file in `raw_dir`.

    Prefers v2 files, then most recently modified.

    Raises
    ------
    MinerUError
        If no matching file exists.
    """
    candidates = list(raw_dir.rglob("*content_list*.json"))

    if not candidates:
        raise MinerUError(
            f"No *content_list*.json file found in {raw_dir}."
        )

    candidates.sort(
        key=lambda path: (
            "v2" in path.name,
            -path.stat().st_mtime,
        )
    )

    return candidates[0]