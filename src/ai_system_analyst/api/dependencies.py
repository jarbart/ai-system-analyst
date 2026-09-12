from pathlib import Path

from ai_system_analyst.llm.ollama_provider import OllamaProvider
from ai_system_analyst.retrieval.hybrid_search import HybridSearch
from ai_system_analyst.retrieval.keyword_search import KeywordSearch
from ai_system_analyst.retrieval.local_embeddings import LocalEmbeddingProvider
from ai_system_analyst.retrieval.reranker import Reranker
from ai_system_analyst.retrieval.simple_reranker import SimpleReranker
from ai_system_analyst.retrieval.vector_store import VectorStore
from ai_system_analyst.services.context_assembler import ContextAssembler
from ai_system_analyst.services.incident_analysis_service import (
    IncidentAnalysisService,
)
from ai_system_analyst.services.indexing_service import IndexingService
from ai_system_analyst.services.knowledge_base import load_knowledge_base


def build_analysis_service() -> IncidentAnalysisService:
    embedding_provider = LocalEmbeddingProvider()

    vector_store = VectorStore(
        dimension=384,
    )

    keyword_search = KeywordSearch()

    hybrid_search = HybridSearch(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        keyword_search=keyword_search,
    )

    indexing_service = IndexingService(hybrid_search)

    load_knowledge_base(
        Path("data/knowledge"),
        indexing_service,
    )

    reranker: Reranker = SimpleReranker()

    context_assembler = ContextAssembler()

    llm_provider = OllamaProvider()

    return IncidentAnalysisService(
        hybrid_search=hybrid_search,
        reranker=reranker,
        context_assembler=context_assembler,
        llm_provider=llm_provider,
    )