from typing import cast

from sentence_transformers import SentenceTransformer

from ai_system_analyst.retrieval.embeddings import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )

        return cast(list[float], embedding.tolist())
