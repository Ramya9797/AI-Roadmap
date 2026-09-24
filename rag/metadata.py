from typing import Dict, List
from pathlib import Path


def get_document_type(source: str) -> str:
    """
    Determine the document type from the source filename.
    """

    filename = Path(source).name.lower()

    if "policy" in filename:
        return "policy"

    if "support" in filename:
        return "support"

    if "strategy" in filename:
        return "strategy"

    return "general"


def get_document_name(source: str) -> str:
    """
    Return the document name without extension.
    """

    return Path(source).stem


def add_metadata(chunks: List[Dict]) -> List[Dict]:
    """
    Add metadata to each document chunk.

    Metadata includes:

    - source
    - document_name
    - document_type
    - chunk_id
    - chunk_index
    - total_chunks
    """

    # First, count how many chunks belong to each document.
    document_chunk_counts = {}

    for chunk in chunks:
        source = chunk["source"]

        document_chunk_counts[source] = (
            document_chunk_counts.get(source, 0) + 1
        )

    enriched_chunks = []

    # Track the current chunk index for each document.
    document_chunk_indexes = {}

    for chunk in chunks:
        enriched_chunk = dict(chunk)

        source = chunk["source"]

        # Existing metadata
        enriched_chunk["document_type"] = get_document_type(source)
        enriched_chunk["document_name"] = get_document_name(source)

        # New metadata
        chunk_index = document_chunk_indexes.get(source, 0)

        enriched_chunk["chunk_index"] = chunk_index
        enriched_chunk["total_chunks"] = document_chunk_counts[source]

        document_chunk_indexes[source] = chunk_index + 1

        enriched_chunks.append(enriched_chunk)

    return enriched_chunks