from dataclasses import dataclass

import faiss
import numpy as np

from ai_system_analyst.domain.models import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        vector = np.asarray([embedding], dtype=np.float32)

        if vector.shape[1] != self._index.d:
            raise ValueError("embedding dimension does not match index dimension")

        self._index.add(vector)
        self._chunks.append(chunk)

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self._chunks:
            return []

        vector = np.asarray([embedding], dtype=np.float32)

        if vector.shape[1] != self._index.d:
            raise ValueError("embedding dimension does not match index dimension")

        scores, indices = self._index.search(
            vector,
            min(top_k, len(self._chunks)),
        )

        return [
            SearchResult(
                chunk=self._chunks[index],
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
