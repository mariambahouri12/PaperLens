# domain/value_objects/ids.py
"""
Typed identifiers. Using `NewType`-like aliases keeps signatures
readable while remaining plain `str` at runtime (no serialization
overhead, easy JSON round-tripping).
"""
from __future__ import annotations

import uuid
from typing import NewType

DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)
ImageId = NewType("ImageId", str)
TableId = NewType("TableId", str)


def new_document_id() -> DocumentId:
    """Generate a fresh, globally unique document id (UUID v4)."""
    return DocumentId(str(uuid.uuid4()))