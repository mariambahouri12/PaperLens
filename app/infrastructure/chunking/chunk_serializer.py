# app/infrastructure/chunking/chunk_serializer.py
"""
Serialize a list of Chunk to JSON.

Used for debugging: the chunking step is otherwise entirely in-memory,
and this module makes its output inspectable without touching the
vector store or the BM25 index.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.domain.entities.chunk import Chunk


def chunk_to_dict(chunk: Chunk) -> dict:
    """Flat, JSON-serializable view of a chunk."""
    meta = chunk.metadata

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": meta.title, 
        "filename": meta.filename,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": {
            "chunk_index": meta.chunk_index,
            "chunk_types": [t.value for t in meta.chunk_types],
            "section_path": meta.section_path.as_list(),
            "page_numbers": list(meta.page_numbers),
            "image_id": meta.image_id,
            "image_path": meta.image_path,
        },
    }


def chunks_to_json(chunks: list[Chunk], *, indent: int = 2) -> str:
    """Return the JSON representation of a list of chunks."""
    return json.dumps(
        [chunk_to_dict(c) for c in chunks],
        ensure_ascii=False,
        indent=indent,
    )


def write_chunks(
    chunks: list[Chunk],
    path: Path,
    *,
    indent: int = 2,
) -> None:
    """Atomically write the chunks to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(chunks_to_json(chunks, indent=indent), encoding="utf-8")
    temporary.replace(path)