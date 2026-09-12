from abc import ABC, abstractmethod

from ai_system_analyst.domain.models import Chunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert text into an embedding vector."""
        raise NotImplementedError

    def embed_chunk(self, chunk: Chunk) -> list[float]:
        """Convert a document chunk into an embedding vector."""
        return self.embed(chunk.content)
