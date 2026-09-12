from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.retrieval.embeddings import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_embedding_provider_embeds_text() -> None:
    provider = FakeEmbeddingProvider()

    embedding = provider.embed("hello")

    assert embedding == [5.0, 1.0]


def test_embedding_provider_can_embed_chunk() -> None:
    provider = FakeEmbeddingProvider()

    chunk = Chunk(
        chunk_id="DOC-001-0000",
        document_id="DOC-001",
        source_type=SourceType.DOCUMENTATION,
        content="hello",
        position=0,
    )

    embedding = provider.embed_chunk(chunk)

    assert embedding == [5.0, 1.0]
