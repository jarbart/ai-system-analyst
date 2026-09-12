from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from ai_system_analyst.api.dependencies import build_analysis_service
from ai_system_analyst.services.incident_analysis_service import (
    IncidentAnalysisService,
)


class AnalyzeRequest(BaseModel):
    query: str


class EvidenceResponse(BaseModel):
    document_id: str
    chunk_id: str
    source_type: str
    content: str
    score: float


class AnalyzeResponse(BaseModel):
    analysis: str
    evidence: list[EvidenceResponse]


app = FastAPI(
    title="AI System Analyst",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_analysis_service() -> IncidentAnalysisService:
    return build_analysis_service()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    service: IncidentAnalysisService = Depends(get_analysis_service),  # noqa: B008
) -> AnalyzeResponse:
    result = service.analyze(request.query)

    return AnalyzeResponse(
        analysis=result.analysis,
        evidence=[
            EvidenceResponse(
                document_id=evidence.chunk.document_id,
                chunk_id=evidence.chunk.chunk_id,
                source_type=evidence.chunk.source_type.value,
                content=evidence.chunk.content,
                score=evidence.score,
            )
            for evidence in result.evidence
        ],
    )