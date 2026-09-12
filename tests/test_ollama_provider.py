import pytest

from ai_system_analyst.llm.models import IncidentAnalysis
from ai_system_analyst.llm.ollama_provider import OllamaProvider


@pytest.mark.integration
def test_ollama_provider_generates_response() -> None:
    provider = OllamaProvider()

    response = provider.generate(
        """
    Analyze this incident and return an incident analysis.

    Incident:
    The Orders API returns HTTP 500 when PostgreSQL is unavailable.

    Return a JSON object with exactly these fields:
    - root_cause: string
    - confirmed_evidence: list of strings
    - hypotheses: list of strings
    - next_steps: list of strings
    """
    )

    assert isinstance(response, IncidentAnalysis)
    assert response.root_cause
    assert isinstance(response.confirmed_evidence, list)
    assert isinstance(response.hypotheses, list)
    assert isinstance(response.next_steps, list)
