from pathlib import Path

from ai_system_analyst.domain.models import SourceType
from ai_system_analyst.ingestion.text_loader import load_text_file
from ai_system_analyst.services.indexing_service import IndexingService


def load_knowledge_base(
    knowledge_dir: Path,
    indexing_service: IndexingService,
) -> int:
    documents = [
        (
            "orders-api",
            "Orders API",
            SourceType.DOCUMENTATION,
            knowledge_dir / "orders-api.txt",
        ),
        (
        "incident-1001",
        "Incident 1001 - Orders API HTTP 500",
        SourceType.INCIDENT,
        knowledge_dir / "incident-1001.txt",
        ),
    ]

    indexed_chunks = 0

    for document_id, title, source_type, path in documents:
        document = load_text_file(
            path,
            document_id=document_id,
            title=title,
            source_type=source_type,
        )

        indexed_chunks += indexing_service.index_document(document)

    return indexed_chunks
