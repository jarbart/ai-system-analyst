import pytest

from ai_system_analyst.domain.models import Document, SourceType
from ai_system_analyst.ingestion.chunker import chunk_document


def test_document_is_split_into_overlapping_chunks() -> None:
    document = Document(
        document_id="DOC-001",
        title="Test document",
        source_type=SourceType.DOCUMENTATION,
        content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )

    chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 4
    assert chunks[0].chunk_id == "DOC-001-0000"
    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[1].content == "IJKLMNOPQR"
    assert chunks[2].content == "QRSTUVWXYZ"
    assert chunks[3].content == "YZ"
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    assert chunks[2].position == 2
    assert chunks[3].position == 3


def test_invalid_chunk_size_raises_error() -> None:
    document = Document(
        document_id="DOC-001",
        title="Test document",
        source_type=SourceType.DOCUMENTATION,
        content="Some content",
    )

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(document, chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    document = Document(
        document_id="DOC-001",
        title="Test document",
        source_type=SourceType.DOCUMENTATION,
        content="Some content",
    )

    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, chunk_size=10, overlap=10)
