#infrastructure/logging/structured_logger.py
"""
Structured logging configuration for PaperLens.

Console:
    Human-readable logs for local development and execution.

File (optional):
    One JSON record per line, suitable for log shipping and aggregation.
    Structured data attached with ``extra={"ctx_<name>": value}`` ends up
    under the ``context`` key of the JSON record.

All application loggers should use the ``paperlens.<layer>`` namespace.

Query tracing
-------------
``QueryTrace`` gathers everything worth knowing about one question:

    - the stages and their durations (retrieval, filtering, LLM, ...)
    - which chunks were sent to the LLM (numbers only, never the text)
    - the LLM statistics reported by Ollama
    - CPU / RAM / GPU usage while the LLM is generating
    - how much of the Ollama model is placed on the GPU
    - citation health (missing or unresolved [S#] markers)

GPU monitoring uses ``pynvml`` (``pip install nvidia-ml-py``) when
available and falls back to the ``nvidia-smi`` command.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import psutil


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        context: dict[str, Any] = {}

        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                context[key[4:]] = value

        if context:
            payload["context"] = context

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure PaperLens logging.

    Calling this function multiple times replaces the previous PaperLens
    handlers instead of creating duplicate log entries.
    """
    logger = logging.getLogger("paperlens")

    logger.setLevel(level.upper())

    # Close before removing so repeated calls (e.g. Streamlit reruns)
    # do not leak file descriptors.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        )
    )
    # Records flagged `file_only` carry structured data for the JSON file
    # and would only duplicate the readable lines in the console.
    console_handler.addFilter(
        lambda record: not getattr(record, "file_only", False)
    )
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

_MAX_QUERY_CHARS = 120
_MAX_SECTION_CHARS = 60
_NS_PER_SECOND = 1_000_000_000

_MARKER_GROUP = re.compile(r"\[([^\[\]]*)\]")
_ONLY_MARKERS = re.compile(r"^\s*S\d+(?:\s*[|,;]\s*S\d+)*\s*$")
_UNRESOLVED = re.compile(r"\[\s*S\d+[^\[\]]*\]")


def _ctx(**fields: Any) -> dict[str, Any]:
    """Build the ``extra`` dict understood by JSONFormatter."""
    return {f"ctx_{key}": value for key, value in fields.items()}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict-like or attribute-style object."""
    if obj is None:
        return default

    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    return default if value is None else value


def _fmt(value: float | None, template: str, missing: str = "n/a") -> str:
    return missing if value is None else template.format(value)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ----------------------------------------------------------------------
# GPU access
# ----------------------------------------------------------------------


class _GpuReader:
    """Read GPU utilisation and VRAM usage (NVML first, nvidia-smi next)."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.total_mb: float | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._smi: str | None = None

        self._init_nvml()

        if self._nvml is None:
            self._smi = shutil.which("nvidia-smi")

    def _init_nvml(self) -> None:
        try:
            import pynvml  # provided by the `nvidia-ml-py` package

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            name = pynvml.nvmlDeviceGetName(handle)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2

            self._nvml = pynvml
            self._handle = handle
        except Exception:
            self._nvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        return self._nvml is not None or self._smi is not None

    def read(self) -> tuple[float, float] | None:
        """Return (utilisation %, VRAM used in MB) or None."""
        if self._nvml is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
                return float(util), used / 1024**2
            except Exception:
                return None

        if self._smi is not None:
            try:
                output = subprocess.run(
                    [
                        self._smi,
                        "--query-gpu=utilization.gpu,memory.used,"
                        "memory.total,name",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).stdout.strip().splitlines()[0]

                util, used, total, name = (
                    part.strip() for part in output.split(",", 3)
                )
                self.name = self.name or name
                self.total_mb = self.total_mb or float(total)
                return float(util), float(used)
            except Exception:
                return None

        return None


_GPU_READER: _GpuReader | None = None


def _gpu_reader() -> _GpuReader:
    global _GPU_READER

    if _GPU_READER is None:
        _GPU_READER = _GpuReader()

    return _GPU_READER


# ----------------------------------------------------------------------
# Resource monitor
# ----------------------------------------------------------------------


class ResourceMonitor:
    """
    Sample CPU, RAM and GPU in a background thread while a block runs.

    CPU values are percentages of the WHOLE machine (0-100). "ollama"
    values only cover processes whose name contains "ollama". GPU values
    describe the whole GPU, not only Ollama.

    Usage:
        with ResourceMonitor() as monitor:
            ...
        monitor.summary
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.summary: dict[str, Any] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._procs: dict[int, psutil.Process] = {}

        self._cpu: list[float] = []
        self._ram: list[float] = []
        self._ollama_cpu: list[float] = []
        self._ollama_rss: list[int] = []
        self._gpu_util: list[float] = []
        self._vram: list[float] = []

    # -- context manager ------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        psutil.cpu_percent(interval=None)  # prime the counter
        self._refresh_ollama()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self._cpu:
            self._sample()  # block shorter than one interval

        self.summary = self._summarise()

        return False

    # -- sampling -------------------------------------------------------

    def _run(self) -> None:
        tick = 0

        while not self._stop.wait(self.interval):
            tick += 1

            if tick % 4 == 0:
                self._refresh_ollama()

            self._sample()

    def _refresh_ollama(self) -> None:
        current: dict[int, psutil.Process] = {}

        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()

            if "ollama" not in name:
                continue

            known = self._procs.get(proc.pid)

            if known is None:
                try:
                    proc.cpu_percent(interval=None)  # prime
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                known = proc

            current[proc.pid] = known

        self._procs = current

    def _sample(self) -> None:
        cores = psutil.cpu_count() or 1

        self._cpu.append(psutil.cpu_percent(interval=None))
        self._ram.append(psutil.virtual_memory().percent)

        ollama_cpu = 0.0
        ollama_rss = 0

        for proc in list(self._procs.values()):
            try:
                ollama_cpu += proc.cpu_percent(interval=None)
                ollama_rss += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self._ollama_cpu.append(ollama_cpu / cores)
        self._ollama_rss.append(ollama_rss)

        gpu = _gpu_reader().read()

        if gpu is not None:
            self._gpu_util.append(gpu[0])
            self._vram.append(gpu[1])

    # -- summary --------------------------------------------------------

    @staticmethod
    def _avg(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _max(values: Sequence[float]) -> float | None:
        return max(values) if values else None

    def _summarise(self) -> dict[str, Any]:
        gpu = _gpu_reader()

        return {
            "samples": len(self._cpu),
            "cpu_avg_pct": self._avg(self._cpu),
            "cpu_max_pct": self._max(self._cpu),
            "ram_max_pct": self._max(self._ram),
            "ollama_cpu_avg_pct": self._avg(self._ollama_cpu),
            "ollama_cpu_max_pct": self._max(self._ollama_cpu),
            "ollama_ram_max_gb": (
                max(self._ollama_rss) / 1024**3 if self._ollama_rss else None
            ),
            "gpu_name": gpu.name,
            "gpu_util_avg_pct": self._avg(self._gpu_util),
            "gpu_util_max_pct": self._max(self._gpu_util),
            "vram_used_max_mb": self._max(self._vram),
            "vram_total_mb": gpu.total_mb,
        }


def format_resources(summary: dict[str, Any]) -> list[str]:
    """Human-readable lines describing a ResourceMonitor summary."""
    lines = [
        "CPU   : machine avg {} | max {} | RAM max {}".format(
            _fmt(summary.get("cpu_avg_pct"), "{:.0f}%"),
            _fmt(summary.get("cpu_max_pct"), "{:.0f}%"),
            _fmt(summary.get("ram_max_pct"), "{:.0f}%"),
        ),
        "Ollama: CPU avg {} | max {} | RAM max {}".format(
            _fmt(summary.get("ollama_cpu_avg_pct"), "{:.0f}%"),
            _fmt(summary.get("ollama_cpu_max_pct"), "{:.0f}%"),
            _fmt(summary.get("ollama_ram_max_gb"), "{:.1f} GB"),
        ),
    ]

    if summary.get("gpu_util_avg_pct") is None:
        lines.append(
            "GPU   : not readable (pip install nvidia-ml-py, "
            "or check that nvidia-smi works)"
        )
    else:
        used = summary.get("vram_used_max_mb")
        total = summary.get("vram_total_mb")

        lines.append(
            "GPU   : {} | util avg {} | max {} | VRAM max {} / {}".format(
                summary.get("gpu_name") or "unknown",
                _fmt(summary.get("gpu_util_avg_pct"), "{:.0f}%"),
                _fmt(summary.get("gpu_util_max_pct"), "{:.0f}%"),
                _fmt(None if used is None else used / 1024, "{:.1f} GB"),
                _fmt(None if total is None else total / 1024, "{:.1f} GB"),
            )
        )

    return lines


# ----------------------------------------------------------------------
# Ollama model placement (what `ollama ps` shows)
# ----------------------------------------------------------------------


def ollama_placement() -> list[dict[str, Any]] | None:
    """
    Return, for each loaded Ollama model, the share placed on the GPU.

    100% GPU is what we want; anything lower means layers run on the CPU.
    Returns None when the Ollama server cannot be queried.
    """
    try:
        import ollama

        response = ollama.ps()
    except Exception:
        return None

    placements: list[dict[str, Any]] = []

    for model in _get(response, "models", []) or []:
        size = _get(model, "size", 0) or 0
        in_vram = _get(model, "size_vram", 0) or 0

        placements.append(
            {
                "model": _get(model, "model") or _get(model, "name"),
                "size_gb": size / 1024**3,
                "vram_gb": in_vram / 1024**3,
                "gpu_pct": (100.0 * in_vram / size) if size else None,
            }
        )

    return placements


# ----------------------------------------------------------------------
# Query trace
# ----------------------------------------------------------------------


class QueryTrace:
    """
    Collect and log everything about the processing of one query.

    Chunk numbering: ``S<n>`` is the marker the LLM sees (position in the
    context, starting at 1) and ``chunk #<i>`` is the chunk_index stored in
    the chunk metadata (the number visible in the *_chunking.json dump).
    """

    def __init__(self, logger: logging.Logger, query_text: str) -> None:
        self.logger = logger
        self.query_id = uuid.uuid4().hex[:8]
        self.stages: dict[str, float] = {}
        self._start = time.perf_counter()

        logger.info(
            "[%s] QUERY: %s",
            self.query_id,
            _shorten(query_text, _MAX_QUERY_CHARS),
            extra=_ctx(query_id=self.query_id, query=query_text),
        )

    # -- stages ---------------------------------------------------------

    @contextmanager
    def stage(self, name: str, *, resources: bool = False) -> Iterator[None]:
        """Time a stage; optionally record CPU/RAM/GPU usage during it."""
        monitor = ResourceMonitor() if resources else None
        start = time.perf_counter()

        if monitor is not None:
            monitor.__enter__()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start

            if monitor is not None:
                monitor.__exit__(None, None, None)

            self.stages[name] = elapsed

            self.logger.info(
                "[%s] %-10s %.2fs",
                self.query_id,
                name,
                elapsed,
                extra=_ctx(
                    query_id=self.query_id,
                    stage=name,
                    seconds=round(elapsed, 3),
                ),
            )

            if monitor is not None:
                self._log_resources(name, monitor.summary)

    def _log_resources(self, stage: str, summary: dict[str, Any]) -> None:
        for line in format_resources(summary):
            self.logger.info("[%s]            %s", self.query_id, line)

        self.logger.info(
            "[%s] resources during %s",
            self.query_id,
            stage,
            extra={
                "file_only": True,
                **_ctx(
                    query_id=self.query_id,
                    stage=stage,
                    resources=summary,
                ),
            },
        )

    # -- chunks sent to the LLM ----------------------------------------

    def log_chunks(
        self,
        chunks: Sequence[Any],
        scores: Sequence[float] | None = None,
        candidates: int | None = None,
    ) -> None:
        """Log which chunks are sent to the LLM (numbers, never the text)."""
        total_tokens = sum(int(chunk.token_count or 0) for chunk in chunks)

        header = "[{}] CONTEXT: {} chunk(s) sent to the LLM, {} tokens".format(
            self.query_id,
            len(chunks),
            total_tokens,
        )

        if candidates is not None:
            header += f" ({candidates} candidate(s) retrieved)"

        self.logger.info(header)

        details: list[dict[str, Any]] = []

        for position, chunk in enumerate(chunks):
            meta = chunk.metadata
            score = scores[position] if scores is not None else None
            pages = ",".join(str(page) for page in meta.page_numbers) or "?"
            types = "+".join(t.value for t in meta.chunk_types)
            section = _shorten(
                meta.section_path.as_string(),
                _MAX_SECTION_CHARS,
            )

            self.logger.info(
                "[%s]   S%-2d | chunk #%-3s | score %s | p.%s | %s | %d tok | %s",
                self.query_id,
                position + 1,
                meta.chunk_index,
                _fmt(score, "{:.4f}"),
                pages,
                types,
                chunk.token_count,
                section,
            )

            details.append(
                {
                    "marker": f"S{position + 1}",
                    "chunk_index": meta.chunk_index,
                    "chunk_id": str(chunk.chunk_id),
                    "score": score,
                    "pages": list(meta.page_numbers),
                    "types": [t.value for t in meta.chunk_types],
                    "tokens": chunk.token_count,
                    "section": meta.section_path.as_string(),
                }
            )

        self.logger.info(
            "[%s] context chunks",
            self.query_id,
            extra={
                "file_only": True,
                **_ctx(
                    query_id=self.query_id,
                    candidates=candidates,
                    context_tokens=total_tokens,
                    chunks=details,
                ),
            },
        )

    # -- LLM ------------------------------------------------------------

    def log_llm(self, stats: dict[str, Any] | None) -> None:
        """Log the statistics reported by Ollama and the GPU placement."""
        if stats:
            self.logger.info(
                "[%s] LLM    : load %.1fs | prompt %d tok in %.1fs | "
                "generated %d tok in %.1fs (%.1f tok/s)",
                self.query_id,
                stats.get("load_s", 0.0),
                stats.get("prompt_tokens", 0),
                stats.get("prompt_s", 0.0),
                stats.get("gen_tokens", 0),
                stats.get("gen_s", 0.0),
                stats.get("tokens_per_s", 0.0),
                extra=_ctx(query_id=self.query_id, llm=stats),
            )
        else:
            self.logger.info(
                "[%s] LLM    : no statistics available from the adapter",
                self.query_id,
            )

        placements = ollama_placement()

        if placements is None:
            self.logger.info(
                "[%s] Ollama : placement unavailable (server not reachable)",
                self.query_id,
            )
            return

        for item in placements:
            gpu_pct = item["gpu_pct"]

            verdict = (
                "all on GPU"
                if gpu_pct is not None and gpu_pct >= 99.5
                else "PARTLY ON CPU (slow)"
            )

            self.logger.info(
                "[%s] Ollama : %s | %s of %s on GPU (%s)",
                self.query_id,
                item["model"],
                _fmt(gpu_pct, "{:.0f}%"),
                _fmt(item["size_gb"], "{:.1f} GB"),
                verdict,
                extra=_ctx(query_id=self.query_id, placement=item),
            )

    # -- citations ------------------------------------------------------

    def log_citations(
        self,
        raw_answer: str,
        final_answer: str,
        chunks: Sequence[Any],
        cited_chunks: Sequence[Any],
    ) -> None:
        """Log which markers the LLM used and warn about citation problems."""
        markers: set[int] = set()

        for group in _MARKER_GROUP.findall(raw_answer or ""):
            if _ONLY_MARKERS.match(group):
                markers.update(int(n) for n in re.findall(r"\d+", group))

        cited_indexes = [c.metadata.chunk_index for c in cited_chunks]

        self.logger.info(
            "[%s] CITED  : markers %s -> chunk #%s",
            self.query_id,
            ["S%d" % n for n in sorted(markers)] or "none",
            cited_indexes or "none",
            extra=_ctx(
                query_id=self.query_id,
                markers=sorted(markers),
                cited_chunk_indexes=cited_indexes,
            ),
        )

        if not markers:
            self.logger.warning(
                "[%s] The LLM answer contains no [S#] marker; no source "
                "can be attached",
                self.query_id,
            )

        unknown = sorted(n for n in markers if n < 1 or n > len(chunks))

        if unknown:
            self.logger.warning(
                "[%s] The LLM cited markers that do not exist: %s",
                self.query_id,
                ["S%d" % n for n in unknown],
            )

        leftovers = _UNRESOLVED.findall(final_answer or "")

        if leftovers:
            self.logger.warning(
                "[%s] Unresolved citation marker(s) left in the final "
                "answer: %s",
                self.query_id,
                leftovers,
            )

    # -- end ------------------------------------------------------------

    def finish(self) -> None:
        total = time.perf_counter() - self._start

        breakdown = " | ".join(
            f"{name} {seconds:.2f}s" for name, seconds in self.stages.items()
        )

        self.logger.info(
            "[%s] TOTAL  : %.2fs (%s)",
            self.query_id,
            total,
            breakdown,
            extra=_ctx(
                query_id=self.query_id,
                total_seconds=round(total, 3),
                stages={k: round(v, 3) for k, v in self.stages.items()},
            ),
        )