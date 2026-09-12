from pathlib import Path

from ai_system_analyst.retrieval.hybrid_search import HybridSearch
from ai_system_analyst.services.indexing_service import IndexingService
from ai_system_analyst.services.knowledge_base import load_knowledge_base


class FakeEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_chunk(self, chunk) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks = []

    def add(self, chunk, embedding) -> None:
        self.chunks.append(chunk)


class FakeKeywordSearch:
    def __init__(self) -> None:
        self.chunks = []

    def add(self, chunk) -> None:
        self.chunks.append(chunk)


def test_load_knowledge_base_indexes_documents(tmp_path: Path) -> None:
    knowledge_file = tmp_path / "orders-api.txt"
    knowledge_file.write_text(
        "POST /api/orders returns HTTP 201.",
        encoding="utf-8",
    )

    incident_file = tmp_path / "incident-1001.txt"
    incident_file.write_text(
        "Orders API returns HTTP 500 because of database connection timeouts.",
        encoding="utf-8",
    )

    vector_store = FakeVectorStore()
    keyword_search = FakeKeywordSearch()

    hybrid_search = HybridSearch(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        keyword_search=keyword_search,
    )

    indexing_service = IndexingService(hybrid_search)

    indexed_chunks = load_knowledge_base(
        tmp_path,
        indexing_service,
    )

    assert indexed_chunks == 2
    assert len(vector_store.chunks) == 2
    assert vector_store.chunks[0].document_id == "orders-api"
