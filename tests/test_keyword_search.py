from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.keyword_search import KeywordSearch


def make_chunk(
    chunk_id: str,
    content: str,
    source_type: SourceType = SourceType.DOCUMENTATION,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="DOC-001",
        source_type=source_type,
        content=content,
        position=0,
    )


def test_keyword_search_returns_matching_chunk() -> None:
    search = KeywordSearch()

    search.add(
        make_chunk(
            "orders",
            "POST /api/orders returns HTTP 500 when database is unavailable.",
        )
    )
    search.add(
        make_chunk(
            "payments",
            "POST /api/payments returns HTTP 400 for invalid payment data.",
        )
    )

    results = search.search("HTTP 500 orders", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "orders"
    assert results[0].score > 0


def test_keyword_search_can_filter_by_source_type() -> None:
    search = KeywordSearch()

    search.add(
        make_chunk(
            "docs",
            "Orders API documentation",
            SourceType.DOCUMENTATION,
        )
    )
    search.add(
        make_chunk(
            "incident",
            "Orders API HTTP 500 incident",
            SourceType.INCIDENT,
        )
    )

    results = search.search(
        "Orders API",
        source_type=SourceType.INCIDENT,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "incident"
