from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    DOCUMENTATION = "documentation"
    INCIDENT = "incident"
    LOG = "log"
    REQUIREMENT = "requirement"
    API_SPECIFICATION = "api_specification"


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    source_type: SourceType
    content: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    position: int
