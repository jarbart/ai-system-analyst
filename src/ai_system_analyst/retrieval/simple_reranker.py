from ai_system_analyst.retrieval.hybrid_search import HybridSearchResult
from ai_system_analyst.retrieval.reranker import Reranker


class SimpleReranker(Reranker):
    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int,
    ) -> list[HybridSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_terms = set(query.lower().split())

        scored_results: list[tuple[float, HybridSearchResult]] = []

        for result in results:
            content_terms = set(result.chunk.content.lower().split())
            matching_terms = query_terms & content_terms

            lexical_score = len(matching_terms) / max(len(query_terms), 1)
            final_score = result.score + lexical_score

            scored_results.append((final_score, result))

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in scored_results[:top_k]
        ]