from abc import ABC, abstractmethod

from ai_system_analyst.retrieval.hybrid_search import HybridSearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int,
    ) -> list[HybridSearchResult]:
        """Rerank retrieved results for a specific query."""
        raise NotImplementedError