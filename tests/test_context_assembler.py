from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.hybrid_search import HybridSearchResult
from ai_system_analyst.services.context_assembler import ContextAssembler


def make_result(
    chunk_id: str,
    content: str,
    score: float = 0.5,
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


def test_context_assembler_includes_source_metadata() -> None:
    assembler = ContextAssembler()

    results = [
        make_result(
            "orders-0001",
            "Orders API returns HTTP 500.",
        )
    ]

    context = assembler.assemble(results)

    assert "[SOURCE: documentation]" in context
    assert "[DOCUMENT: doc-orders-0001]" in context
    assert "[CHUNK: orders-0001]" in context
    assert "Orders API returns HTTP 500." in context


def test_context_assembler_separates_multiple_chunks() -> None:
    assembler = ContextAssembler()

    results = [
        make_result("chunk-1", "First piece of evidence."),
        make_result("chunk-2", "Second piece of evidence."),
    ]

    context = assembler.assemble(results)

    assert "First piece of evidence." in context
    assert "Second piece of evidence." in context
    assert "\n\n---\n\n" in context


def test_context_assembler_respects_max_chars() -> None:
    assembler = ContextAssembler()

    results = [
        make_result("chunk-1", "First piece of evidence."),
        make_result("chunk-2", "Second piece of evidence."),
    ]

    context = assembler.assemble(
        results,
        max_chars=150,
    )

    assert "First piece of evidence." in context
    assert "Second piece of evidence." not in context


def test_context_assembler_rejects_invalid_max_chars() -> None:
    assembler = ContextAssembler()

    try:
        assembler.assemble([], max_chars=0)
    except ValueError as error:
        assert str(error) == "max_chars must be greater than 0"
    else:
        raise AssertionError("Expected ValueError")
