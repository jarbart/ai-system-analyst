from ai_system_analyst.domain.models import Chunk
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.vector_store import SearchResult, VectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def add_chunk(self, chunk: Chunk) -> None:
        embedding = self._embedding_provider.embed_chunk(chunk)
        self._vector_store.add(chunk, embedding)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        embedding = self._embedding_provider.embed(query)

        return self._vector_store.search(
            embedding,
            top_k=top_k,
        )
