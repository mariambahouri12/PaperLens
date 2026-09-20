"""
Typed identifiers. Using `NewType`-like aliases keeps signatures
readable while remaining plain `str` at runtime (no serialization
overhead, easy JSON round-tripping).
"""
from __future__ import annotations

from typing import NewType

DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)
ImageId = NewType("ImageId", str)
TableId = NewType("TableId", str)