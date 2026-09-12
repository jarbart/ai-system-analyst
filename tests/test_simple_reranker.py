from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.hybrid_search import HybridSearchResult
from ai_system_analyst.retrieval.simple_reranker import SimpleReranker


def make_result(
    chunk_id: str,
    content: str,
    score: float,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            source_type=SourceType.DOCUMENTATION,
            content=content,
            position=0,
        ),
        score=score,
    )


def test_simple_reranker_prefers_query_term_matches() -> None:
    reranker = SimpleReranker()

    results = [
        make_result(
            "generic",
            "The service processes requests.",
            score=0.20,
        ),
        make_result(
            "orders",
            "Orders API returns HTTP 500.",
            score=0.15,
        ),
    ]

    reranked = reranker.rerank(
        query="Orders API HTTP 500",
        results=results,
        top_k=1,
    )

    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == "orders"