from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.vector_store import VectorStore
from ai_system_analyst.services.retrieval_service import RetrievalService


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        if "orders" in text.lower():
            return [1.0, 0.0, 0.0]

        if "payments" in text.lower():
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="DOC-001",
        source_type=SourceType.DOCUMENTATION,
        content=content,
        position=0,
    )


def test_retrieval_service_indexes_and_searches_chunks() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = VectorStore(dimension=3)

    service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    service.add_chunk(
        make_chunk("chunk-orders", "Orders API returns HTTP 500")
    )
    service.add_chunk(
        make_chunk("chunk-payments", "Payments API returns HTTP 400")
    )

    results = service.search("Orders API problem", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-orders"
