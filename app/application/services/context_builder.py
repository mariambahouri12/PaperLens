"""
Build the LLM context string from retrieved chunks.

Keeps chunk boundaries visible (labeled by chunk_id and section path)
so the LLM can cite them, and appends image/table captions so the
model has the visual context without needing the binary payload.
"""
from __future__ import annotations

from app.domain.entities.chunk import Chunk


def build_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        header = f"[chunk_id={c.chunk_id} | section={c.metadata.section_path.as_string() or 'root'}]"
        block = [header, c.text]
        if c.metadata.image_captions:
            block.append("Figure captions: " + " | ".join(c.metadata.image_captions))
        if c.metadata.table_captions:
            block.append("Table captions: " + " | ".join(c.metadata.table_captions))
        parts.append("\n".join(block))
    return "\n\n---\n\n".join(parts)