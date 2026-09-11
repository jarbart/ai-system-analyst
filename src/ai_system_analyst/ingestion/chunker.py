from ai_system_analyst.domain.models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must not be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    position = 0

    while start < len(document.content):
        end = start + chunk_size
        content = document.content[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}-{position:04d}",
                document_id=document.document_id,
                content=content,
                position=position,
            )
        )

        position += 1
        start += chunk_size - overlap

    return chunks
