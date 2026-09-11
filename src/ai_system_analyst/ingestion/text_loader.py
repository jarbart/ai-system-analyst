from pathlib import Path

from ai_system_analyst.domain.models import Document, SourceType


def load_text_file(
    path: Path,
    document_id: str,
    title: str,
    source_type: SourceType,
) -> Document:
    content = path.read_text(encoding="utf-8")

    return Document(
        document_id=document_id,
        title=title,
        source_type=source_type,
        content=content,
    )
