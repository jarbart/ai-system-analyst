from pathlib import Path

from ai_system_analyst.domain.models import SourceType
from ai_system_analyst.ingestion.text_loader import load_text_file


def test_load_text_file(tmp_path: Path) -> None:
    source_file = tmp_path / "incident.txt"
    source_file.write_text(
        "Orders started failing after deployment 2.4.",
        encoding="utf-8",
    )

    document = load_text_file(
        path=source_file,
        document_id="INC-001",
        title="Order API returns HTTP 500",
        source_type=SourceType.INCIDENT,
    )

    assert document.document_id == "INC-001"
    assert document.title == "Order API returns HTTP 500"
    assert document.source_type == SourceType.INCIDENT
    assert document.content == "Orders started failing after deployment 2.4."
