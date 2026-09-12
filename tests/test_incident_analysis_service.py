from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.llm.provider import LLMProvider
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider
from ai_system_analyst.retrieval.hybrid_search import HybridSearch, HybridSearchResult
from ai_system_analyst.retrieval.keyword_search import KeywordSearch
from ai_system_analyst.retrieval.reranker import Reranker
from ai_system_analyst.retrieval.vector_store import VectorStore
from ai_system_analyst.services.context_assembler import ContextAssembler
from ai_system_analyst.services.incident_analysis_service import (
    IncidentAnalysisService,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeReranker(Reranker):
    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int,
    ) -> list[HybridSearchResult]:
        return results[:top_k]


class FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Fake incident analysis"


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="orders-0001",
        document_id="orders-doc",
        source_type=SourceType.DOCUMENTATION,
        content="Orders API returns HTTP 500 when PostgreSQL is unavailable.",
        position=0,
    )


def test_incident_analysis_service_builds_prompt_and_calls_llm() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = VectorStore(dimension=3)
    keyword_search = KeywordSearch()

    hybrid_search = HybridSearch(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        keyword_search=keyword_search,
    )

    hybrid_search.add_chunk(make_chunk())

    llm_provider = FakeLLMProvider()

    service = IncidentAnalysisService(
        hybrid_search=hybrid_search,
        reranker=FakeReranker(),
        context_assembler=ContextAssembler(),
        llm_provider=llm_provider,
    )

    result = service.analyze(
        "Orders API returns HTTP 500",
        top_k=1,
    )

    assert result == "Fake incident analysis"
    assert "Orders API returns HTTP 500" in llm_provider.last_prompt
    assert "PostgreSQL is unavailable" in llm_provider.last_prompt
    assert "Likely root cause" in llm_provider.last_prompt
    assert "Recommended next steps" in llm_provider.last_prompt
