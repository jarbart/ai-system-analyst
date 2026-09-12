from ai_system_analyst.domain.models import Document, SourceType
from ai_system_analyst.services.indexing_service import IndexingService


class FakeHybridSearch:
    def __init__(self) -> None:
        self.chunks = []

    def add_chunk(self, chunk) -> None:
        self.chunks.append(chunk)


def test_index_document_adds_all_chunks() -> None:
    document = Document(
        document_id="doc-001",
        title="Orders API",
        source_type=SourceType.DOCUMENTATION,
        content="A" * 1200,
    )

    hybrid_search = FakeHybridSearch()
    service = IndexingService(hybrid_search)

    indexed_count = service.index_document(
        document,
        chunk_size=500,
        overlap=100,
    )

    assert indexed_count == 3
    assert len(hybrid_search.chunks) == 3
    assert hybrid_search.chunks[0].document_id == "doc-001"
