from ai_system_analyst.domain.models import Chunk, Document, SourceType


def test_document_can_be_created() -> None:
    document = Document(
        document_id="INC-001",
        title="Order API returns HTTP 500",
        source_type=SourceType.INCIDENT,
        content="Orders started failing after deployment 2.4.",
    )

    assert document.document_id == "INC-001"
    assert document.source_type == SourceType.INCIDENT


def test_chunk_can_be_created() -> None:
    chunk = Chunk(
        chunk_id="INC-001-0001",
        document_id="INC-001",
        source_type=SourceType.INCIDENT,
        content="Orders started failing after deployment 2.4.",
        position=0,
    )

    assert chunk.chunk_id == "INC-001-0001"
    assert chunk.document_id == "INC-001"
    assert chunk.source_type == SourceType.INCIDENT
    assert chunk.position == 0
