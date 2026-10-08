"""
Mapper from persisted retrieval payloads to domain Chunk entities.

The vector store and BM25 index return flat dictionaries because
persistent indexes should not depend on domain object serialization.

This mapper is the application-layer boundary between those payloads
and the domain model.
"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.entities.chunk import Chunk, ChunkMetadata, ChunkType
from app.domain.exceptions import RetrievalError
from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath

logger = logging.getLogger("paperlens.application.mapper")


class ChunkPayloadMapper:
    """Reconstruct domain Chunk entities from persisted payloads."""

    def to_chunk(
        self,
        chunk_id: str,
        payload: dict[str, Any],
    ) -> Chunk:
        if not isinstance(payload, dict):
            raise RetrievalError(
                f"Invalid payload for chunk '{chunk_id}': "
                "expected a dictionary"
            )

        try:
            text = payload["text"]
            document_id = payload["document_id"]
            token_count = payload["token_count"]
        except KeyError as exc:
            raise RetrievalError(
                f"Missing required field '{exc.args[0]}' "
                f"in payload for chunk '{chunk_id}'"
            ) from exc

        if not isinstance(text, str):
            raise RetrievalError(
                f"Invalid text for chunk '{chunk_id}'"
            )

        if not isinstance(document_id, str) or not document_id:
            raise RetrievalError(
                f"Invalid document_id for chunk '{chunk_id}'"
            )

        try:
            token_count = int(token_count)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Invalid token_count for chunk '{chunk_id}'"
            ) from exc

        if token_count < 0:
            raise RetrievalError(
                f"token_count cannot be negative for chunk '{chunk_id}'"
            )

        metadata = ChunkMetadata(
            document_id=DocumentId(document_id),
            filename=str(payload.get("filename") or ""),
            chunk_index=self._parse_int(
                payload.get("chunk_index", 0),
                "chunk_index",
                chunk_id,
            ),
            chunk_types=self._parse_types(
                payload.get("types", "")
            ),
            section_path=SectionPath.of(
                payload.get("section_path") or []
            ),
            page_numbers=self._parse_page_numbers(
                payload.get("page_numbers") or [],
                chunk_id,
            ),
            image_id=self._optional_image_id(
                payload.get("image_id")
            ),
            image_path=(
                str(payload["image_path"])
                if payload.get("image_path")
                else None
            ),
            table_id=self._optional_table_id(
                payload.get("table_id")
            ),
        )

        return Chunk(
            chunk_id=ChunkId(chunk_id),
            document_id=metadata.document_id,
            text=text,
            metadata=metadata,
            token_count=token_count,
        )

    @staticmethod
    def _parse_int(
        value: Any,
        field_name: str,
        chunk_id: str,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Invalid {field_name} for chunk '{chunk_id}'"
            ) from exc

        if parsed < 0:
            raise RetrievalError(
                f"{field_name} cannot be negative for chunk '{chunk_id}'"
            )

        return parsed

    @staticmethod
    def _parse_page_numbers(
        value: Any,
        chunk_id: str,
    ) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)):
            raise RetrievalError(
                f"Invalid page_numbers for chunk '{chunk_id}'"
            )

        pages: list[int] = []

        for page in value:
            try:
                page_number = int(page)
            except (TypeError, ValueError) as exc:
                raise RetrievalError(
                    f"Invalid page number for chunk '{chunk_id}'"
                ) from exc

            if page_number < 0:
                raise RetrievalError(
                    f"Page numbers cannot be negative "
                    f"for chunk '{chunk_id}'"
                )

            pages.append(page_number)

        return tuple(pages)

    @staticmethod
    def _parse_types(
        raw: Any,
    ) -> tuple[ChunkType, ...]:
        if isinstance(raw, (list, tuple)):
            names = raw
        else:
            names = str(raw or "").split(",")

        parsed: list[ChunkType] = []

        for name in names:
            normalized = str(name).strip()

            if not normalized:
                continue

            try:
                parsed.append(ChunkType(normalized))
            except ValueError:
                logger.warning(
                    "Unknown chunk type in persisted payload: %r",
                    normalized,
                )

        return tuple(parsed)

    @staticmethod
    def _optional_image_id(
        value: Any,
    ) -> ImageId | None:
        if value is None or value == "":
            return None

        return ImageId(str(value))

    @staticmethod
    def _optional_table_id(
        value: Any,
    ) -> TableId | None:
        if value is None or value == "":
            return None

        return TableId(str(value))