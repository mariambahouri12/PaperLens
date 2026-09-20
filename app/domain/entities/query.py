from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    TEXT_AND_IMAGE = "text_and_image"


@dataclass
class Query:
    text: str
    intent: QueryIntent = QueryIntent.TEXT_ONLY
    top_k: int | None = None