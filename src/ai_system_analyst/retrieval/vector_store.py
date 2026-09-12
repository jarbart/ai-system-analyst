from dataclasses import dataclass

import faiss
import numpy as np

from ai_system_analyst.domain.models import Chunk, SourceType


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
        source_type: SourceType | None = None,
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
            len(self._chunks),
        )

        results: list[SearchResult] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self._chunks[index]

            if source_type is not None and chunk.source_type != source_type:
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

            if len(results) >= top_k:
                break

        return results
