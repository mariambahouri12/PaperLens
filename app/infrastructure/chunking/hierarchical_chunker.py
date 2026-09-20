"""
Two-stage hierarchical chunking.

Stage 1 — Structure-aware:
    Sections whose token size is <= SECTION_MAX_TOKENS become a single
    chunk that preserves the *full* section path.

Stage 2 — Fixed-size fallback:
    Sections larger than SECTION_MAX_TOKENS are split into fixed-size
    windows of CHUNK_SIZE tokens with CHUNK_OVERLAP tokens of overlap.
    Every resulting sub-chunk keeps the same section path and inherits
    the section's image/table references (so figures near the top of a
    section are still retrievable from any of its sub-chunks).

Tables are never split in the middle: a table always ends up in a
single chunk with its caption, and the chunk text includes a Markdown
rendering of the table so the LLM sees its structure.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.config.settings import Settings
from app.domain.entities.chunk import Chunk, ChunkMetadata
from app.domain.entities.document import Document
from app.domain.entities.element import DocumentElement, ElementType
from app.domain.repositories.chunker import ChunkerPort
from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.chunking.token_counter import count_tokens

logger = logging.getLogger("paperlens.infrastructure.chunking")


class HierarchicalChunker(ChunkerPort):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ #
    def chunk(self, document: Document) -> list[Chunk]:
        groups = self._group_by_section(document)
        chunks: list[Chunk] = []

        for section_path, elements in groups.items():
            images, tables = self._assets_for(document, section_path, elements)
            text = self._render_section_text(elements, tables)
            if not text.strip():
                continue

            tokens = count_tokens(text)
            if tokens <= self.settings.section_max_tokens:
                chunks.append(self._make_chunk(
                    document, section_path, text, elements, images, tables,
                ))
            else:
                for i, (sub_text, sub_elements) in enumerate(self._fixed_size_windows(elements, tables)):
                    chunks.append(self._make_chunk(
                        document, section_path, sub_text, sub_elements, images, tables,
                        suffix=f"s{i}",
                    ))

        logger.info("Chunked %s into %d chunk(s)", document.metadata.document_id, len(chunks))
        return chunks

    # ------------------------------------------------------------------ #
    def _group_by_section(self, document: Document) -> dict[SectionPath, list[DocumentElement]]:
        groups: dict[SectionPath, list[DocumentElement]] = defaultdict(list)
        for e in document.elements:
            if e.element_type == ElementType.HEADING:
                continue
            groups[e.section_path].append(e)
        return groups

    def _render_section_text(
        self,
        elements: list[DocumentElement],
        tables: list,
    ) -> str:
        parts: list[str] = []
        table_by_id = {t.table_id: t for t in tables}
        for e in sorted(elements, key=lambda x: x.order):
            if e.element_type == ElementType.TABLE and e.table_id in table_by_id:
                tbl = table_by_id[e.table_id]
                block = [f"[Table {e.table_id}]"]
                if tbl.caption:
                    block.append(tbl.caption)
                block.append(tbl.to_markdown())
                parts.append("\n".join(block))
            elif e.text:
                parts.append(e.text)
        return "\n\n".join(parts).strip()

    def _assets_for(
        self,
        document: Document,
        section_path: SectionPath,
        elements: list[DocumentElement],
    ):
        images = [img for img in document.images if img.section_path == section_path]
        tables = [tbl for tbl in document.tables if tbl.section_path == section_path]
        # Also include assets explicitly referenced by elements of this section.
        referenced_image_ids = {e.image_id for e in elements if e.image_id}
        referenced_table_ids = {e.table_id for e in elements if e.table_id}
        for img in document.images:
            if img.image_id in referenced_image_ids and img not in images:
                images.append(img)
        for tbl in document.tables:
            if tbl.table_id in referenced_table_ids and tbl not in tables:
                tables.append(tbl)
        return images, tables

    def _fixed_size_windows(
        self,
        elements: list[DocumentElement],
        tables: list,
    ) -> list[tuple[str, list[DocumentElement]]]:
        """Split a large section into windows of CHUNK_SIZE tokens with
        CHUNK_OVERLAP tokens overlap. Table elements are never split."""
        windows: list[tuple[str, list[DocumentElement]]] = []
        current_texts: list[str] = []
        current_elements: list[DocumentElement] = []
        current_tokens = 0
        table_by_id = {t.table_id: t for t in tables}

        def flush() -> None:
            nonlocal current_texts, current_elements, current_tokens
            if current_texts:
                windows.append(("\n\n".join(current_texts), current_elements))
            # Prepare overlap from the tail of what was just flushed.
            overlap_texts: list[str] = []
            overlap_tokens = 0
            for t in reversed(current_texts):
                tk = count_tokens(t)
                if overlap_tokens + tk > self.settings.chunk_overlap:
                    break
                overlap_texts.insert(0, t)
                overlap_tokens += tk
            current_texts = list(overlap_texts)
            current_elements = []
            current_tokens = overlap_tokens

        for e in sorted(elements, key=lambda x: x.order):
            if e.element_type == ElementType.TABLE and e.table_id in table_by_id:
                rendered = table_by_id[e.table_id].to_markdown()
            else:
                rendered = e.text
            if not rendered.strip():
                continue
            rendered_tokens = count_tokens(rendered)
            if current_tokens + rendered_tokens > self.settings.chunk_size and current_texts:
                flush()
            current_texts.append(rendered)
            current_elements.append(e)
            current_tokens += rendered_tokens

        if current_texts:
            windows.append(("\n\n".join(current_texts), current_elements))

        return windows

    def _make_chunk(
        self,
        document: Document,
        section_path: SectionPath,
        text: str,
        elements: list[DocumentElement],
        images: list,
        tables: list,
        suffix: str = "",
    ) -> Chunk:
        pages = sorted({e.page_number for e in elements if e.page_number})
        image_ids = [ImageId(img.image_id) for img in images]
        image_captions = [img.caption for img in images if img.caption]
        table_ids = [TableId(t.table_id) for t in tables]
        table_captions = [t.caption for t in tables if t.caption]

        chunk_id = ChunkId(
            f"{document.metadata.document_id}_{'_'.join(p[:3] for p in section_path.parts) or 'root'}"
            f"_{suffix or 'full'}"
        )

        meta = ChunkMetadata(
            document_id=DocumentId(document.metadata.document_id),
            filename=document.metadata.filename,
            section_path=section_path,
            page_numbers=pages,
            image_ids=image_ids,
            image_captions=image_captions,
            table_ids=table_ids,
            table_captions=table_captions,
        )
        return Chunk(
            chunk_id=chunk_id,
            document_id=meta.document_id,
            text=text,
            metadata=meta,
            token_count=count_tokens(text),
        )