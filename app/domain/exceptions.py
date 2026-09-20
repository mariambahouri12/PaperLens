"""Domain-level exceptions. Infrastructure layers raise these; library-
specific exceptions must not leak past their own boundary."""


class PaperLensError(Exception):
    """Base class for every error raised by PaperLens."""


class ExtractionError(PaperLensError):
    """Raised when a document cannot be parsed or its structure recovered."""


class ChunkingError(PaperLensError):
    """Raised when a document cannot be turned into retrieval chunks."""


class EmbeddingError(PaperLensError):
    """Raised when embeddings cannot be produced."""


class IndexingError(PaperLensError):
    """Raised when the vector store or BM25 index cannot be written."""


class RetrievalError(PaperLensError):
    """Raised when retrieval fails at any stage."""


class ImageStoreError(PaperLensError):
    """Raised when an image cannot be stored or loaded."""


class LLMError(PaperLensError):
    """Raised when the local LLM fails to produce an answer."""