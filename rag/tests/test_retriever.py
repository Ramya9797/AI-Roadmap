import pytest

from rag.retriever import Retriever
from rag.vector_store import ChromaVectorStore


def create_store():
    store = ChromaVectorStore(
        collection_name="retriever_unit_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
        },
        {
            "source": "shipping.md",
            "text": "Standard shipping takes 5 business days.",
            "chunk_id": "shipping.md:0",
            "document_type": "shipping",
            "document_name": "shipping",
        },
        {
            "source": "support.md",
            "text": "Contact support for account assistance.",
            "chunk_id": "support.md:0",
            "document_type": "support",
            "document_name": "support",
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add_chunks(chunks, embeddings)

    return store


def test_retriever_returns_top_k():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=2,
    )

    result = retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    assert len(result["ids"][0]) == 2


def test_retriever_returns_most_similar_document():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=1,
    )

    result = retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    assert result["ids"][0][0] == "refund_policy.md:0"


def test_retriever_returns_document_text():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=1,
    )

    result = retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    assert (
        result["documents"][0][0]
        == "Refunds are available within 30 days."
    )


def test_retriever_returns_metadata():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=1,
    )

    result = retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    metadata = result["metadatas"][0][0]

    assert metadata["source"] == "refund_policy.md"


def test_retriever_top_k_one():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=1,
    )

    result = retriever.retrieve(
        [0.0, 1.0, 0.0]
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "shipping.md:0"


def test_retriever_top_k_three():
    store = create_store()

    retriever = Retriever(
        store,
        top_k=3,
    )

    result = retriever.retrieve(
        [0.0, 1.0, 0.0]
    )

    assert len(result["ids"][0]) == 3


def test_invalid_top_k():
    store = create_store()

    with pytest.raises(ValueError):
        Retriever(
            store,
            top_k=0,
        )


def test_retriever_empty_store():
    store = ChromaVectorStore(
        collection_name="empty_retriever_test"
    )

    retriever = Retriever(
        store,
        top_k=3,
    )

    result = retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    assert result["ids"] == [[]]