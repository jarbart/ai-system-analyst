from pathlib import Path

from ai_system_analyst.domain.models import SourceType
from ai_system_analyst.ingestion.chunker import chunk_document
from ai_system_analyst.ingestion.text_loader import load_text_file
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.vector_store import VectorStore
from ai_system_analyst.services.retrieval_service import RetrievalService


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        text_lower = text.lower()

        if "orders" in text_lower:
            return [1.0, 0.0, 0.0]

        if "payments" in text_lower:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def test_rag_pipeline_loads_chunks_and_retrieves_relevant_content(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "api_docs.txt"

    document_path.write_text(
        """
Orders API

The Orders API creates and retrieves customer orders.
HTTP 500 may occur when the order database is unavailable.

Payments API

The Payments API processes customer payments.
HTTP 400 indicates invalid payment data.
""".strip(),
        encoding="utf-8",
    )

    document = load_text_file(
        path=document_path,
        document_id="API-DOCS",
        title="API Documentation",
        source_type=SourceType.DOCUMENTATION,
    )

    chunks = chunk_document(
        document,
        chunk_size=120,
        overlap=20,
    )

    embedding_provider = FakeEmbeddingProvider()
    vector_store = VectorStore(dimension=3)

    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    for chunk in chunks:
        retrieval_service.add_chunk(chunk)

    results = retrieval_service.search(
        "Orders API HTTP 500",
        top_k=2,
    )

    assert len(results) == 2
    assert "Orders API" in results[0].chunk.content
