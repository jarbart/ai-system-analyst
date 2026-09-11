from ai_system_analyst.domain.models import Document, SourceType


def test_document_can_be_created() -> None:
    document = Document(
        document_id="INC-001",
        title="Order API returns HTTP 500",
        source_type=SourceType.INCIDENT,
        content="Orders started failing after deployment 2.4.",
    )

    assert document.document_id == "INC-001"
    assert document.source_type == SourceType.INCIDENT
