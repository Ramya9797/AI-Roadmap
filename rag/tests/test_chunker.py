import pytest

from rag.chunker import chunk_text, chunk_documents


def test_chunk_text_returns_multiple_chunks():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) > 1


def test_chunk_size_is_respected():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    for chunk in chunks:
        assert len(chunk) <= 10


def test_chunk_overlap_is_preserved():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks[0][-2:] == chunks[1][:2]


def test_small_document_returns_one_chunk():
    text = "small document"

    chunks = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks == ["small document"]


def test_empty_text_returns_no_chunks():
    chunks = chunk_text(
        "",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks == []


def test_whitespace_text_returns_no_chunks():
    chunks = chunk_text(
        "     ",
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "hello",
            chunk_size=0,
            chunk_overlap=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "hello",
            chunk_size=10,
            chunk_overlap=10,
        )


def test_chunk_documents_preserves_source():
    documents = [
        {
            "source": "refund_policy.md",
            "text": "abcdefghijklmnopqrstuvwxyz",
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    for chunk in chunks:
        assert chunk["source"] == "refund_policy.md"


def test_chunk_documents_creates_chunk_ids():
    documents = [
        {
            "source": "refund_policy.md",
            "text": "abcdefghijklmnopqrstuvwxyz",
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunks[0]["chunk_id"] == "refund_policy.md:0"
    assert chunks[1]["chunk_id"] == "refund_policy.md:1"


def test_chunk_documents_contains_required_fields():
    documents = [
        {
            "source": "refund_policy.md",
            "text": "abcdefghijklmnopqrstuvwxyz",
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    for chunk in chunks:
        assert "source" in chunk
        assert "text" in chunk
        assert "chunk_id" in chunk


def test_chunk_documents_is_deterministic():
    documents = [
        {
            "source": "refund_policy.md",
            "text": "abcdefghijklmnopqrstuvwxyz",
        }
    ]

    first_result = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    second_result = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert first_result == second_result