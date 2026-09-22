import pytest

from rag.metadata import (
    get_document_type,
    get_document_name,
    add_metadata,
)


def test_policy_document_type():
    assert get_document_type("refund_policy.md") == "policy"


def test_support_document_type():
    assert get_document_type("customer_support.md") == "support"


def test_strategy_document_type():
    assert get_document_type("churn_strategy.md") == "strategy"


def test_unknown_document_type():
    assert get_document_type("random_document.md") == "general"


def test_document_name_removes_extension():
    assert get_document_name("refund_policy.md") == "refund_policy"


def test_document_name_for_support():
    assert get_document_name("customer_support.md") == "customer_support"


def test_document_name_for_strategy():
    assert get_document_name("churn_strategy.md") == "churn_strategy"


def test_add_metadata_preserves_source():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        }
    ]

    result = add_metadata(chunks)

    assert result[0]["source"] == "refund_policy.md"


def test_add_metadata_preserves_text():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        }
    ]

    result = add_metadata(chunks)

    assert result[0]["text"] == "Refund information"


def test_add_metadata_preserves_chunk_id():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        }
    ]

    result = add_metadata(chunks)

    assert result[0]["chunk_id"] == "refund_policy.md:0"


def test_add_metadata_creates_document_type():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        }
    ]

    result = add_metadata(chunks)

    assert result[0]["document_type"] == "policy"


def test_add_metadata_creates_document_name():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        }
    ]

    result = add_metadata(chunks)

    assert result[0]["document_name"] == "refund_policy"


def test_add_metadata_does_not_change_chunk_count():
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refund information",
            "chunk_id": "refund_policy.md:0",
        },
        {
            "source": "order_policy.md",
            "text": "Order information",
            "chunk_id": "order_policy.md:0",
        },
    ]

    result = add_metadata(chunks)

    assert len(result) == len(chunks)


def test_add_metadata_handles_empty_list():
    result = add_metadata([])

    assert result == []