import pytest

from ai_system_analyst.retrieval.local_embeddings import LocalEmbeddingProvider


@pytest.mark.integration
def test_local_embedding_provider_returns_vector() -> None:
    provider = LocalEmbeddingProvider()

    embedding = provider.embed("Orders API returns HTTP 500.")

    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)
