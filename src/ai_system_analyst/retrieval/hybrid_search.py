from dataclasses import dataclass

from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.keyword_search import KeywordSearch
from ai_system_analyst.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class HybridSearchResult:
    chunk: Chunk
    score: float


class HybridSearch:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._keyword_search = keyword_search
        self._rrf_k = rrf_k

    def add_chunk(self, chunk: Chunk) -> None:
        embedding = self._embedding_provider.embed_chunk(chunk)

        self._vector_store.add(chunk, embedding)
        self._keyword_search.add(chunk)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: SourceType | None = None,
    ) -> list[HybridSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        embedding = self._embedding_provider.embed(query)

        semantic_results = self._vector_store.search(
            embedding,
            top_k=top_k,
            source_type=source_type,
        )

        keyword_results = self._keyword_search.search(
            query,
            top_k=top_k,
            source_type=source_type,
        )

        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        for rank, result in enumerate(semantic_results, start=1):
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0.0) + (
                1.0 / (self._rrf_k + rank)
            )
            chunks[result.chunk.chunk_id] = result.chunk

        for rank, keyword_result in enumerate(keyword_results, start=1):
            scores[keyword_result.chunk.chunk_id] = scores.get(
                keyword_result.chunk.chunk_id,
                0.0,
            ) + (1.0 / (self._rrf_k + rank))

            chunks[keyword_result.chunk.chunk_id] = keyword_result.chunk

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        return [
            HybridSearchResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
            )
            for chunk_id in ranked_ids[:top_k]
        ]