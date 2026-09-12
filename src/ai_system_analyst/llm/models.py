from pydantic import BaseModel


class IncidentAnalysis(BaseModel):
    root_cause: str
    confirmed_evidence: list[str]
    hypotheses: list[str]
    next_steps: list[str]
