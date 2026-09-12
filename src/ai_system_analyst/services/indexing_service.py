from ai_system_analyst.domain.models import Document
from ai_system_analyst.ingestion.chunker import chunk_document
from ai_system_analyst.retrieval.hybrid_search import HybridSearch


class IndexingService:
    def __init__(
        self,
        hybrid_search: HybridSearch,
    ) -> None:
        self._hybrid_search = hybrid_search

    def index_document(
        self,
        document: Document,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> int:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            self._hybrid_search.add_chunk(chunk)

        return len(chunks)
