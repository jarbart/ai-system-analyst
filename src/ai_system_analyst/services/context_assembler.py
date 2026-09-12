from ai_system_analyst.retrieval.hybrid_search import HybridSearchResult


class ContextAssembler:
    def assemble(
        self,
        results: list[HybridSearchResult],
        max_chars: int = 4000,
    ) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        sections: list[str] = []
        total_chars = 0

        for result in results:
            chunk = result.chunk

            section = (
                f"[SOURCE: {chunk.source_type.value}]\n"
                f"[DOCUMENT: {chunk.document_id}]\n"
                f"[CHUNK: {chunk.chunk_id}]\n"
                f"{chunk.content.strip()}"
            )

            if total_chars + len(section) > max_chars:
                break

            sections.append(section)
            total_chars += len(section)

        return "\n\n---\n\n".join(sections)
