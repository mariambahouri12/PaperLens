"""
Hybrid chunker working on the domain Document.

For each section (in reading order), consecutive blocks are grouped by
nature:

    text / equation  -> merged together (an equation stays with the text
                        below it), packed up to max_tokens with overlap
    list             -> merged, packed item by item
    table            -> one chunk per table (caption + Markdown grid)
    image / chart    -> one chunk per caption, file path in metadata

Blocks of different nature are never merged, and two text runs separated
by a table or an image stay in separate chunks. Chunks never cross a
section boundary, so section / subsection metadata is always exact.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from app.domain.entities.block import (
    Block,
    EquationBlock,
    ListBlock,
    TableBlock,
    TextBlock,
    VisualBlock,
)
from app.domain.entities.chunk import Chunk, ChunkMetadata, ChunkType
from app.domain.entities.document import Document
from app.domain.repositories.chunker import ChunkerPort
from app.domain.value_objects.ids import ChunkId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.chunking.config import ChunkingConfig
from app.infrastructure.chunking.drafts import ChunkDraft
from app.infrastructure.chunking.renderers import render_table, render_visual
from app.infrastructure.chunking.text_packer import Unit, make_unit, pack_units
from app.infrastructure.chunking.token_counter import count_tokens

logger = logging.getLogger("paperlens.infrastructure.chunking")


class Kind(str, Enum):
    TEXT = "text"  # text + equation
    LIST = "list"
    TABLE = "table"
    VISUAL = "visual"


# Kinds whose consecutive blocks can be merged into one chunk.
MERGEABLE = {Kind.TEXT, Kind.LIST}


@dataclass
class Run:
    """Consecutive blocks of the same kind within one section."""

    kind: Kind
    blocks: list[Block] = field(default_factory=list)


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------


def classify(block: Block) -> Kind | None:
    if isinstance(block, (TextBlock, EquationBlock)):
        return Kind.TEXT

    if isinstance(block, ListBlock):
        return Kind.LIST

    if isinstance(block, TableBlock):
        return Kind.TABLE

    if isinstance(block, VisualBlock):
        return Kind.VISUAL

    return None


def group_runs(blocks: Iterable[Block]) -> list[Run]:
    runs: list[Run] = []

    for block in blocks:
        kind = classify(block)

        if kind is None:
            continue

        if runs and kind in MERGEABLE and runs[-1].kind == kind:
            runs[-1].blocks.append(block)
        else:
            runs.append(Run(kind=kind, blocks=[block]))

    return runs


# ----------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------


def _group_equations_with_text(blocks: list[Block]) -> list[list[Block]]:
    """
    Group blocks so that every equation travels with the text below it.
    Equations at the very end of the run stay with the text above.
    """
    groups: list[list[Block]] = []
    pending: list[Block] = []

    for block in blocks:
        if isinstance(block, EquationBlock):
            pending.append(block)
            continue

        groups.append(pending + [block])
        pending = []

    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)

    return groups


def _render_text_block(block: Block) -> str:
    return block.latex if isinstance(block, EquationBlock) else block.text


def build_text_units(blocks: list[Block]) -> list[Unit]:
    units = []

    for group in _group_equations_with_text(blocks):
        text = "\n\n".join(filter(None, map(_render_text_block, group)))

        if not text:
            continue

        types = {ChunkType.TEXT}

        if any(isinstance(block, EquationBlock) for block in group):
            types.add(ChunkType.EQUATION)

        units.append(make_unit(text, (block.page for block in group), types))

    return units


def build_list_units(blocks: list[Block]) -> list[Unit]:
    return [
        make_unit(item, [block.page], {ChunkType.LIST})
        for block in blocks
        for item in block.items
    ]


def _pages(block: Block) -> tuple[int, ...]:
    return (block.page,) if block.page else ()


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------


class HierarchicalChunker(ChunkerPort):
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        """All chunks share the document's `document_id`."""
        chunks: list[Chunk] = []

        for section in document.iter_sections():
            for run in group_runs(section.blocks):
                for draft in self._drafts_for(run):
                    chunks.append(
                        self._build_chunk(
                            document,
                            section.path,
                            draft,
                            index=len(chunks),
                        )
                    )

        logger.info(
            "Chunked %s into %d chunk(s)",
            document.metadata.document_id,
            len(chunks),
        )

        return chunks

    # ------------------------------------------------------------------

    def _drafts_for(self, run: Run) -> list[ChunkDraft]:
        if run.kind == Kind.TEXT:
            return pack_units(build_text_units(run.blocks), self.config)

        if run.kind == Kind.LIST:
            return pack_units(
                build_list_units(run.blocks),
                self.config,
                separator="\n",
            )

        # Table and visual runs always contain exactly one block.
        block = run.blocks[0]

        if run.kind == Kind.TABLE:
            return self._table_drafts(block)

        return self._visual_drafts(block)

    def _table_drafts(self, table: TableBlock) -> list[ChunkDraft]:
        text = render_table(table)

        if not text:
            return []

        if count_tokens(text) > self.config.max_tokens:
            logger.warning(
                "Table %s exceeds max_tokens and is kept whole",
                table.table_id,
            )

        return [
            ChunkDraft(
                text=text,
                types=(ChunkType.TABLE,),
                pages=_pages(table),
                image_path=table.image_path,
                table_id=table.table_id,
            )
        ]

    def _visual_drafts(self, visual: VisualBlock) -> list[ChunkDraft]:
        text = render_visual(visual)

        if not text:
            logger.debug("Skipping visual without caption (page %s)", visual.page)
            return []

        return [
            ChunkDraft(
                text=text,
                types=(ChunkType(visual.kind.value),),
                pages=_pages(visual),
                image_id=visual.image_id,
                image_path=visual.image_path,
            )
        ]

    @staticmethod
    def _build_chunk(
        document: Document,
        section_path: SectionPath,
        draft: ChunkDraft,
        index: int,
    ) -> Chunk:
        document_id = document.metadata.document_id

        return Chunk(
            chunk_id=ChunkId(f"{document_id}_{index:04d}"),
            document_id=document_id,
            text=draft.text,
            token_count=count_tokens(draft.text),
            metadata=ChunkMetadata(
                document_id=document_id,
                filename=document.metadata.filename,
                chunk_index=index,
                chunk_types=draft.types,
                section_path=section_path,
                page_numbers=draft.pages,
                image_id=draft.image_id,
                image_path=draft.image_path,
                table_id=draft.table_id,
            ),
        )