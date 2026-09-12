from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.hybrid_search import HybridSearch
from ai_system_analyst.retrieval.keyword_search import KeywordSearch
from ai_system_analyst.retrieval.vector_store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        if "Orders" in text or "orders" in text:
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]

def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_type=SourceType.DOCUMENTATION,
        content=content,
        position=0,
    )


def test_hybrid_search_combines_semantic_and_keyword_results() -> None:
    provider = FakeEmbeddingProvider()
    vector_store = VectorStore(dimension=3)
    keyword_search = KeywordSearch()

    hybrid_search = HybridSearch(
        embedding_provider=provider,
        vector_store=vector_store,
        keyword_search=keyword_search,
    )

    hybrid_search.add_chunk(
        make_chunk(
            "orders",
            "Orders API returns HTTP 500 when PostgreSQL is unavailable.",
        )
    )

    hybrid_search.add_chunk(
        make_chunk(
            "payments",
            "Payments API returns HTTP 400 for invalid payment data.",
        )
    )

    results = hybrid_search.search(
        "Orders API HTTP 500",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "orders"