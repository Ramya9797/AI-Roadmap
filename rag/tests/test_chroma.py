from rag.vector_store import ChromaVectorStore


def test_chroma_collection_created():
    store = ChromaVectorStore(
        collection_name="chroma_collection_test"
    )

    assert store.collection is not None
    assert store.collection.name == "chroma_collection_test"


def test_documents_are_indexed():
    store = ChromaVectorStore(
        collection_name="chroma_index_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "shipping.md",
            "text": "Standard shipping takes 5 business days.",
            "chunk_id": "shipping.md:0",
            "document_type": "shipping",
            "document_name": "shipping",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    assert store.count() == 2



def test_similarity_search():
    store = ChromaVectorStore(
        collection_name="chroma_similarity_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "shipping.md",
            "text": "Standard shipping takes 5 business days.",
            "chunk_id": "shipping.md:0",
            "document_type": "shipping",
            "document_name": "shipping",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "refund_policy.md:0"


def test_metadata_persisted():
    store = ChromaVectorStore(
        collection_name="chroma_metadata_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        }
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
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
    assert metadata["total_chunks"] == 1


def test_reingestion_is_idempotent():
    store = ChromaVectorStore(
        collection_name="chroma_idempotent_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "shipping.md",
            "text": "Standard shipping takes 5 business days.",
            "chunk_id": "shipping.md:0",
            "document_type": "shipping",
            "document_name": "shipping",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    first_count = store.count()

    store.add_chunks(
        chunks,
        embeddings,
    )

    second_count = store.count()

    assert first_count == 2
    assert second_count == 2



def test_metadata_filter_policy():
    store = ChromaVectorStore(
        collection_name="chroma_policy_filter_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "support.md",
            "text": "Contact support for account assistance.",
            "chunk_id": "support.md:0",
            "document_type": "support",
            "document_name": "support",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    store.add_chunks(
        chunks,
        embeddings,
    )

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        where={"document_type": "policy"},
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "refund_policy.md:0"
    assert (
        result["metadatas"][0][0]["document_type"]
        == "policy"
    )


def test_metadata_filter_support():
    store = ChromaVectorStore(
        collection_name="chroma_support_filter_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "support.md",
            "text": "Contact support for account assistance.",
            "chunk_id": "support.md:0",
            "document_type": "support",
            "document_name": "support",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "strategy.md",
            "text": "Our strategy focuses on customer growth.",
            "chunk_id": "strategy.md:0",
            "document_type": "strategy",
            "document_name": "strategy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        where={"document_type": "support"},
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "support.md:0"
    assert result["metadatas"][0][0]["document_type"] == "support"


def test_metadata_filter_strategy():
    store = ChromaVectorStore(
        collection_name="chroma_strategy_filter_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "strategy.md",
            "text": "Our strategy focuses on customer growth.",
            "chunk_id": "strategy.md:0",
            "document_type": "strategy",
            "document_name": "strategy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        where={"document_type": "strategy"},
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "strategy.md:0"
    assert result["metadatas"][0][0]["document_type"] == "strategy"


def test_metadata_filter_unknown_returns_empty():
    store = ChromaVectorStore(
        collection_name="chroma_unknown_filter_test"
    )

    chunks = [
        {
            "source": "refund_policy.md",
            "text": "Refunds are available within 30 days.",
            "chunk_id": "refund_policy.md:0",
            "document_type": "policy",
            "document_name": "refund_policy",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "source": "support.md",
            "text": "Contact support for account assistance.",
            "chunk_id": "support.md:0",
            "document_type": "support",
            "document_name": "support",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        where={"document_type": "unknown"},
    )

    assert result["ids"][0] == []
    assert result["documents"][0] == []
    assert result["metadatas"][0] == []