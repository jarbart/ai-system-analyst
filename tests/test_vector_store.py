import pytest

from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.vector_store import VectorStore


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="DOC-001",
        source_type=SourceType.DOCUMENTATION,
        content=content,
        position=0,
    )


def test_vector_store_returns_most_similar_chunk() -> None:
    store = VectorStore(dimension=3)

    store.add(
        make_chunk("chunk-1", "Orders API"),
        [1.0, 0.0, 0.0],
    )
    store.add(
        make_chunk("chunk-2", "Payments API"),
        [0.0, 1.0, 0.0],
    )
    store.add(
        make_chunk("chunk-3", "Authentication"),
        [0.0, 0.0, 1.0],
    )

    results = store.search([0.9, 0.1, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[1].chunk.chunk_id == "chunk-2"
    assert results[0].score > results[1].score


def test_vector_store_rejects_wrong_dimension() -> None:
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError, match="dimension"):
        store.add(
            make_chunk("chunk-1", "Orders API"),
            [1.0, 0.0],
        )


def test_vector_store_returns_empty_for_empty_store() -> None:
    store = VectorStore(dimension=3)

    assert store.search([1.0, 0.0, 0.0]) == []


def test_vector_store_can_filter_by_source_type() -> None:
    store = VectorStore(dimension=3)

    store.add(
        Chunk(
            chunk_id="docs-1",
            document_id="DOC-001",
            source_type=SourceType.DOCUMENTATION,
            content="Orders API documentation",
            position=0,
        ),
        [1.0, 0.0, 0.0],
    )

    store.add(
        Chunk(
            chunk_id="incident-1",
            document_id="INC-001",
            source_type=SourceType.INCIDENT,
            content="Orders API incident",
            position=0,
        ),
        [0.9, 0.1, 0.0],
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=5,
        source_type=SourceType.INCIDENT,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "incident-1"
