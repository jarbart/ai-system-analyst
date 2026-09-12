from ai_system_analyst.llm.provider import LLMProvider
from ai_system_analyst.retrieval.hybrid_search import HybridSearch
from ai_system_analyst.retrieval.reranker import Reranker
from ai_system_analyst.services.context_assembler import ContextAssembler


class IncidentAnalysisService:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: Reranker,
        context_assembler: ContextAssembler,
        llm_provider: LLMProvider,
    ) -> None:
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._context_assembler = context_assembler
        self._llm_provider = llm_provider

    def analyze(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ) -> str:
        retrieved = self._hybrid_search.search(
            query,
            top_k=top_k,
        )

        reranked = self._reranker.rerank(
            query,
            retrieved,
            top_k=top_k,
        )

        context = self._context_assembler.assemble(
            reranked,
            max_chars=max_context_chars,
        )

        prompt = (
            "You are an AI system analyst investigating an incident.\n\n"
            "Use only the provided evidence when forming your analysis.\n"
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Incident/query:\n{query}\n\n"
            f"Evidence:\n{context}\n\n"
            "Provide a concise analysis with:\n"
            "1. Likely root cause\n"
            "2. Supporting evidence\n"
            "3. Recommended next steps\n"
        )

        return self._llm_provider.generate(prompt)
