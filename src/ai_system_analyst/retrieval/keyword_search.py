from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from ai_system_analyst.domain.models import Chunk, SourceType


@dataclass(frozen=True)
class KeywordSearchResult:
    chunk: Chunk
    score: float


class KeywordSearch:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w/-]*\b",
        )
        self._chunks: list[Chunk] = []
        self._matrix = None

    def add(self, chunk: Chunk) -> None:
        self._chunks.append(chunk)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        contents = [chunk.content for chunk in self._chunks]
        self._matrix = self._vectorizer.fit_transform(contents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: SourceType | None = None,
    ) -> list[KeywordSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self._chunks:
            return []

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]

        ranked_indices = scores.argsort()[::-1]

        results: list[KeywordSearchResult] = []

        for index in ranked_indices:
            chunk = self._chunks[index]

            if source_type is not None and chunk.source_type != source_type:
                continue

            results.append(
                KeywordSearchResult(
                    chunk=chunk,
                    score=float(scores[index]),
                )
            )

            if len(results) >= top_k:
                break

        return results
