# domain/exceptions.py
"""
Exception hierarchy for PaperLens.

- `PaperLensError` is the base class.
- Business exceptions (`ExtractionError`, `IndexingError`, ...) are raised
  by the domain and application layers.
- Technical exceptions (`MinerUError`, `GeminiError`) are raised by the
  infrastructure and MUST be translated by the caller if they need to
  cross a layer boundary.
"""
from __future__ import annotations


class PaperLensError(Exception):
    """Base class for all PaperLens errors."""


# ----------------------------------------------------------------------
# Business exceptions (domain / application)
# ----------------------------------------------------------------------


class ExtractionError(PaperLensError):
    """Raised when a document cannot be extracted or mapped."""


class IndexingError(PaperLensError):
    """Raised when the vector store, BM25 or checkpoint fails."""


class EmbeddingError(PaperLensError):
    """Raised when the embedder fails."""


class LLMError(PaperLensError):
    """Raised when the LLM fails."""


class ImageResolverError(PaperLensError):
    """Raised when an image referenced by a chunk cannot be resolved."""


# ----------------------------------------------------------------------
# Technical exceptions (infrastructure internals)
# ----------------------------------------------------------------------


class MinerUError(PaperLensError):
    """Raised when the MinerU subprocess fails or its output is missing."""


class GeminiError(PaperLensError):
    """Raised when the Gemini API is unavailable or returns invalid data."""

class RetrievalError(PaperLensError):
    """Raised when retrieval or retrieval-result processing fails."""


# Backwards-compatible alias (to be removed once all callers migrate).
ImageStoreError = ImageResolverError