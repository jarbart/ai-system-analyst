from fastapi.testclient import TestClient

from ai_system_analyst.api.app import app, get_analysis_service
from ai_system_analyst.domain.models import Chunk, SourceType
from ai_system_analyst.services.incident_analysis_service import (
    Evidence,
    IncidentAnalysisResult,
)


class FakeAnalysisService:
    def analyze(self, query: str) -> IncidentAnalysisResult:
        chunk = Chunk(
            chunk_id="test-0001",
            document_id="test-document",
            source_type=SourceType.INCIDENT,
            content="Test evidence",
            position=0,
        )

        return IncidentAnalysisResult(
            analysis=f"Analysis requested for: {query}",
            evidence=[
                Evidence(
                    chunk=chunk,
                    score=1.0,
                )
            ],
        )


def test_analyze_endpoint_returns_analysis() -> None:
    app.dependency_overrides[get_analysis_service] = (
        lambda: FakeAnalysisService()
    )

    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={"query": "Orders API returns HTTP 500"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "analysis": "Analysis requested for: Orders API returns HTTP 500",
        "evidence": [
            {
                "document_id": "test-document",
                "chunk_id": "test-0001",
                "source_type": "incident",
                "content": "Test evidence",
                "score": 1.0,
            }
        ],
    }
    app.dependency_overrides.clear()


def test_analyze_endpoint_validates_request() -> None:
    app.dependency_overrides[get_analysis_service] = (
        lambda: FakeAnalysisService()
    )

    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={},
    )

    assert response.status_code == 422

    app.dependency_overrides.clear()