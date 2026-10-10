
# config/settings.py
"""
Single source of truth for all configurable parameters.
Nothing in domain/ or application/ should contain a magic number that
belongs here. Everything is read from environment variables (with
sensible defaults), so the same code runs locally, in tests, or in a
container without edits.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERLENS_",
        env_file=".env",
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------
    data_dir: Path = Field(default=Path("./data"))
    image_dir: Path = Field(default=Path("./images"))
    storage_dir: Path = Field(default=Path("./storage"))

    # --- Models --------------------------------------------------------
    llm_model: str = Field(default="qwen3:8b")
    seed: int = Field(default=42)

    # --- Embeddings ----------------------------------------------------
    embedding_model: str = Field(default="nomic-ai/nomic-embed-text-v1.5")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_dimension: int = Field(default=768, ge=1)
    embedding_max_seq_length: int = Field(default=1024, ge=1)
    embedding_device: str | None = Field(default=None)
    embedding_document_prefix: str = Field(default="search_document: ")
    embedding_query_prefix: str = Field(default="search_query: ")
    embedding_normalize: bool = Field(default=True)
    embedding_trust_remote_code: bool = Field(default=True)

    # --- Chunking ------------------------------------------------------
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=80, ge=0)

    # --- Retrieval -----------------------------------------------------
    min_relevance_score: float = Field(default=0.005)
    max_chunks: int = Field(default=6, ge=1)
    max_context_tokens: int = Field(default=6000, ge=1)

    # Retrieval strategies (disabled by default)
    use_bm25: bool = Field(default=False)
    use_dense: bool = Field(default=False)
    use_rrf: bool = Field(default=True)

    rrf_k: int = Field(default=60, ge=1)
    retrieval_candidate_multiplier: int = Field(default=1, ge=1)

     # --- Reranking -----------------------------------------------------
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_candidate_k: int = Field(default=30, ge=1)
    reranker_device: str = Field(default="cpu")

    # --- LLM -----------------------------------------------------------
    temperature: float = Field(default=0.0)
    num_ctx: int = Field(default=8384, ge=1)
    llm_think: bool = Field(default=False)

    # --- Observability -------------------------------------------------
    log_level: str = Field(default="INFO")

    # --- Derived paths -------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        """Directory where PDFs uploaded from the web UI are stored."""
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        """Final artifacts only: `<stem>_chunking.json`."""
        return self.data_dir / "output"

    @property
    def debug_dir(self) -> Path:
        """Intermediate artifacts: extraction JSON, Gemini cache."""
        return self.data_dir / "debug"

    @property
    def mineru_raw_dir(self) -> Path:
        """Raw MinerU output: content_list.json + images."""
        return self.data_dir / "mineru_raw"

    @property
    def vector_store_path(self) -> Path:
        return self.storage_dir / "vector"

    @property
    def bm25_store_path(self) -> Path:
        return self.storage_dir / "bm25"

    @property
    def checkpoint_path(self) -> Path:
        return self.storage_dir / "embedding_checkpoint.json"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.image_dir,
            self.storage_dir,
            self.uploads_dir,
            self.output_dir,
            self.debug_dir,
            self.mineru_raw_dir,
            self.vector_store_path,
            self.bm25_store_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
