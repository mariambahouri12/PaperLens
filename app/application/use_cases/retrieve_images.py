"""
Use case: image-focused retrieval.

Given a query that explicitly asks for a figure/diagram, run the same
hybrid retrieval and return only the images referenced by the top
chunks. This keeps image retrieval aligned with the text pipeline
(same index, same embeddings) without needing a multimodal vector DB.
"""
from __future__ import annotations

import logging

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.domain.entities.image import ExtractedImage
from app.domain.entities.query import Query, QueryIntent

logger = logging.getLogger("paperlens.application.retrieve_images")


class RetrieveImagesUseCase:
    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self.answer_use_case = answer_use_case

    def run(self, query_text: str) -> list[ExtractedImage]:
        query = Query(text=query_text, intent=QueryIntent.IMAGE_ONLY)
        answer = self.answer_use_case.run(query)
        logger.info("Image query returned %d image(s)", len(answer.used_images))
        return answer.used_images