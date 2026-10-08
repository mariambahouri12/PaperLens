# chunking/hierarchical_chunker.py
"""
Hybrid chunker working on the domain Document.

For each section (in reading order), consecutive blocks are grouped by
nature:

    text / equation  -> merged together (an equation stays with the text
                        below it), packed up to max_tokens with overlap
    list             -> merged, packed item by item
    table            -> one chunk per table (caption + Markdown grid);
                        oversized tables are cut between rows, repeating
                        caption, header and footnote in every part
    image / chart    -> one chunk per caption, file path in metadata

Splitting hierarchy for text, from coarsest to finest:

    whole paragraph  ->  sentences  ->  token windows (last resort)

A paragraph is only exploded into sentences when it does not fit in
max_tokens. Sentence splitting never cuts inside math or LaTeX, and a
unit containing an equation is never cut, even if it slightly exceeds
max_tokens.

Blocks of different nature are never merged, and two text runs
separated by a table or an image stay in separate chunks. Chunks never
cross a section boundary, so section / subsection metadata is always
exact.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, TypeVar

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
from app.infrastructure.chunking.renderers import render_table_parts, render_visual
from app.infrastructure.chunking.sentences import split_sentences
from app.infrastructure.chunking.text_packer import Unit, make_unit, pack_units
from app.infrastructure.chunking.token_counter import count_tokens

logger = logging.getLogger("paperlens.infrastructure.chunking")

T = TypeVar("T")


class Kind(str, Enum):
    TEXT = "text"  # text + equation
    LIST = "list"
    TABLE = "table"
    VISUAL = "visual"


# Kinds whose consecutive blocks can be merged into one chunk.
MERGEABLE = {Kind.TEXT, Kind.LIST}


@dataclass
class Run:
    """Consecutive blocks with the same chunking kind within one section."""

    kind: Kind
    blocks: list[Block] = field(default_factory=list)


class Atom(NamedTuple):
    """A sentence or an equation, the finest piece text is split into."""

    text: str
    page: int
    is_equation: bool
    starts_paragraph: bool


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


def _glue_leading(
    items: list[T],
    is_leading: Callable[[T], bool],
) -> list[list[T]]:
    """
    Group items so that every "leading" item (an equation) travels with
    the item below it. Leading items at the very end stay with the item
    above.
    """
    groups: list[list[T]] = []
    pending: list[T] = []

    for item in items:
        if is_leading(item):
            pending.append(item)
            continue

        groups.append(pending + [item])
        pending = []

    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)

    return groups


def _render_text_block(block: Block) -> str:
    return block.latex if isinstance(block, EquationBlock) else block.text


def _text_types(has_equation: bool) -> set[ChunkType]:
    return {ChunkType.TEXT} | ({ChunkType.EQUATION} if has_equation else set())


def build_text_units(blocks: list[Block], config: ChunkingConfig) -> list[Unit]:
    """
    One unit per (equations + paragraph) group. A group that does not
    fit in max_tokens is exploded into sentence-level units.
    """
    units: list[Unit] = []

    for group in _glue_leading(blocks, lambda b: isinstance(b, EquationBlock)):
        text = "\n\n".join(filter(None, map(_render_text_block, group)))

        if not text:
            continue

        if count_tokens(text) <= config.max_tokens:
            has_equation = any(isinstance(b, EquationBlock) for b in group)
            units.append(
                make_unit(text, (b.page for b in group), _text_types(has_equation))
            )
        else:
            units.extend(_explode(group))

    return units


def _explode(group: list[Block]) -> list[Unit]:
    """
    Oversized group -> one unit per sentence. Equations are kept whole
    and glued to the sentence below; any unit with an equation is atomic.
    """
    atoms: list[Atom] = []

    for block in group:
        if isinstance(block, EquationBlock):
            if block.latex:
                atoms.append(Atom(block.latex, block.page, True, True))
            continue

        atoms.extend(
            Atom(sentence, block.page, False, index == 0)
            for index, sentence in enumerate(split_sentences(block.text))
        )

    units: list[Unit] = []

    for sub in _glue_leading(atoms, lambda atom: atom.is_equation):
        has_equation = any(atom.is_equation for atom in sub)
        first = sub[0]

        units.append(
            make_unit(
                "\n\n".join(atom.text for atom in sub),
                (atom.page for atom in sub),
                _text_types(has_equation),
                # Sentences of the same paragraph are joined by a space,
                # anything else starts a new paragraph.
                sep="\n\n" if first.is_equation or first.starts_paragraph else " ",
                atomic=has_equation,
            )
        )

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
            return pack_units(
                build_text_units(run.blocks, self.config),
                self.config,
            )

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
        parts = [
            text
            for text in render_table_parts(table, self.config.max_tokens)
            if text
        ]

        if len(parts) > 1:
            logger.info("Table %s split into %d parts", table.table_id, len(parts))

        for text in parts:
            if count_tokens(text) > self.config.max_tokens:
                logger.warning(
                    "Table %s has a part exceeding max_tokens "
                    "(very wide rows); kept as is",
                    table.table_id,
                )
                break

        return [
            ChunkDraft(
                text=text,
                types=(ChunkType.TABLE,),
                pages=_pages(table),
                image_path=table.image_path,
                table_id=table.table_id,
            )
            for text in parts
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