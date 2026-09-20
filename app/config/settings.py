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
    model_config = SettingsConfigDict(env_prefix="PAPERLENS_", env_file=".env", extra="ignore")

    # --- Paths ---------------------------------------------------------
    data_dir: Path = Field(default=Path("./data"))
    image_dir: Path = Field(default=Path("./images"))
    storage_dir: Path = Field(default=Path("./storage"))

    # --- Models --------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    llm_model: str = Field(default="qwen3:8b")

    # --- Chunking ------------------------------------------------------
    section_max_tokens: int = Field(default=1200)
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=80)

    # --- Retrieval -----------------------------------------------------
    min_relevance_score: float = Field(default=0.005)
    max_chunks: int = Field(default=10)
    max_context_tokens: int = Field(default=6000)
    rrf_k: int = Field(default=60)

    # --- LLM -----------------------------------------------------------
    temperature: float = Field(default=0.0)
    num_ctx: int = Field(default=8384)

    # --- Observability -------------------------------------------------
    log_level: str = Field(default="INFO")

    # --- Derived paths -------------------------------------------------
    @property
    def vector_store_path(self) -> Path:
        return self.storage_dir / "vector"

    @property
    def bm25_store_path(self) -> Path:
        return self.storage_dir / "bm25"

    def ensure_directories(self) -> None:
        for p in (self.data_dir, self.image_dir, self.storage_dir,
                  self.vector_store_path, self.bm25_store_path):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()