import uuid

import pytest

from rag.vector_store import ChromaVectorStore


@pytest.fixture
def store():
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    return ChromaVectorStore(collection_name=collection_name)


def create_test_chunk(
    chunk_id="refund_policy.md:0",
):
    return {
        "source": "refund_policy.md",
        "text": "Refunds are available within 30 days.",
        "chunk_id": chunk_id,
        "document_type": "policy",
        "document_name": "refund_policy",
        "chunk_index": int(chunk_id.split(":")[-1]),
        "total_chunks": 2,
    }


def test_empty_collection(store):
    assert store.count() == 0


def test_add_single_chunk(store):
    chunks = [
        create_test_chunk()
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    assert store.count() == 1


def test_add_multiple_chunks(store):
    chunks = [
        create_test_chunk("refund_policy.md:0"),
        create_test_chunk("refund_policy.md:1"),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    assert store.count() == 2


def test_get_all_returns_ids(store):
    chunks = [
        create_test_chunk()
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    result = store.get_all()

    assert result["ids"] == [
        "refund_policy.md:0"
    ]


def test_get_all_returns_documents(store):
    chunks = [
        create_test_chunk()
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    result = store.get_all()

    assert (
        result["documents"][0]
        == "Refunds are available within 30 days."
    )


def test_metadata_is_stored(store):
    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 5,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    result = store.get_all()

    metadata = result["metadatas"][0]

    assert metadata["source"] == "refund_policy.md"
    assert metadata["document_type"] == "policy"
    assert metadata["document_name"] == "refund_policy"
    assert metadata["chunk_id"] == "refund_policy.md:0"
    assert metadata["chunk_index"] == 0
    assert metadata["total_chunks"] == 5


def test_mismatched_chunk_embedding_count(store):
    chunks = [
        create_test_chunk()
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with pytest.raises(ValueError):
        store.add_chunks(
            chunks,
            embeddings,
        )


def test_empty_input(store):
    store.add_chunks(
        [],
        [],
    )

    assert store.count() == 0