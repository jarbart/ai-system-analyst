from dataclasses import dataclass

from ai_system_analyst.domain.models import Chunk
from ai_system_analyst.llm.provider import LLMProvider
from ai_system_analyst.retrieval.hybrid_search import HybridSearch
from ai_system_analyst.retrieval.reranker import Reranker
from ai_system_analyst.services.context_assembler import ContextAssembler


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IncidentAnalysisResult:
    root_cause: str
    confirmed_evidence: list[str]
    hypotheses: list[str]
    next_steps: list[str]
    evidence: list[Evidence]

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
    ) -> IncidentAnalysisResult:
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
            "You are an AI system analyst investigating a production incident.\n\n"
            "Return ONLY valid JSON with exactly these fields:\n"
            "{\n"
            '  "root_cause": "string",\n'
            '  "confirmed_evidence": ["string"],\n'
            '  "hypotheses": ["string"],\n'
            '  "next_steps": ["string"]\n'
            "}\n\n"
            "IMPORTANT RULES:\n"
            "- Use ONLY the provided evidence.\n"
            "- Every factual claim must be supported by the evidence.\n"
            "- Never invent numbers, configuration values, timestamps, causes, "
            "or system behavior that are not explicitly present in the evidence.\n"
            "- Do not assume that a common industry practice applies to this system.\n"
            "- Distinguish clearly between confirmed facts and hypotheses.\n"
            "- Never present a hypothesis as a confirmed fact.\n"
            "- If the evidence does not prove something, say: "
            "\"The available evidence is insufficient to determine this.\"\n"
            "- When referring to evidence, include its CHUNK identifier.\n\n"
            f"Incident/query:\n{query}\n\n"
            f"Evidence:\n{context}\n"
        )

        analysis = self._llm_provider.generate(prompt)

        return IncidentAnalysisResult(
            root_cause=analysis.root_cause,
            confirmed_evidence=analysis.confirmed_evidence,
            hypotheses=analysis.hypotheses,
            next_steps=analysis.next_steps,
            evidence=[
                Evidence(
                    chunk=result.chunk,
                    score=result.score,
                )
                for result in reranked
            ],
        )
