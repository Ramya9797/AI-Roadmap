from typing import List, Dict


DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of text chunks.
    """

    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Chunk loaded documents while preserving their source.
    """

    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(document_chunks):
            chunks.append(
                {
                    "source": source,
                    "text": chunk,
                    "chunk_id": f"{source}:{index}",
                }
            )

    return chunks