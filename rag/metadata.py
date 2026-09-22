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
    """

    enriched_chunks = []

    for chunk in chunks:
        enriched_chunk = dict(chunk)

        source = chunk["source"]

        enriched_chunk["document_type"] = get_document_type(source)
        enriched_chunk["document_name"] = get_document_name(source)

        enriched_chunks.append(enriched_chunk)

    return enriched_chunks