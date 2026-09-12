import pytest

from ai_system_analyst.llm.ollama_provider import OllamaProvider


@pytest.mark.integration
def test_ollama_provider_generates_response() -> None:
    provider = OllamaProvider()

    response = provider.generate(
        "Explain HTTP 500 in one short sentence."
    )

    assert response.strip()
    assert isinstance(response, str)
