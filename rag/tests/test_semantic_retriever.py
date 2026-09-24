from rag.semantic_retriever import SemanticRetrieverAdapter


class FakeEmbeddingModel:
    def embed_documents(self, documents):
        assert documents == ["refund policy"]
        return [[0.1, 0.2, 0.3]]


class FakeSemanticRetriever:
    def retrieve(self, query_embedding, where=None):
        assert query_embedding == [0.1, 0.2, 0.3]

        return {
            "ids": [["doc1", "doc2"]],
            "documents": [
                [
                    "Refunds are allowed.",
                    "Orders can be cancelled.",
                ]
            ],
            "distances": [[0.0, 1.0]],
        }


def test_semantic_adapter_returns_standard_results():
    retriever = FakeSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    results = adapter.search(
        "refund policy",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "Refunds are allowed."
    assert results[0]["score"] == 1.0


def test_semantic_adapter_converts_distance_to_score():
    retriever = FakeSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    results = adapter.search(
        "refund policy",
        top_k=2,
    )

    assert results[1]["score"] == 0.5



def test_semantic_adapter_rejects_empty_query():
    retriever = FakeSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    try:
        adapter.search("")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "Query cannot be empty"


def test_semantic_adapter_rejects_zero_top_k():
    retriever = FakeSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    try:
        adapter.search("refund policy", top_k=0)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "top_k must be greater than 0"


def test_semantic_adapter_rejects_negative_top_k():
    retriever = FakeSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    try:
        adapter.search("refund policy", top_k=-1)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "top_k must be greater than 0"


def test_semantic_adapter_preserves_metadata():
    class MetadataSemanticRetriever:
        def retrieve(self, query_embedding, where=None):
            return {
                "ids": [["doc1"]],
                "documents": [["Refunds are allowed."]],
                "metadatas": [[
                    {
                        "source": "refund_policy.txt",
                        "document_type": "policy",
                        "document_name": "refund_policy",
                        "chunk_id": "refund_policy.txt:0",
                        "chunk_index": 0,
                        "total_chunks": 2,
                    }
                ]],
                "distances": [[0.0]],
            }

    retriever = MetadataSemanticRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    results = adapter.search(
        "refund policy",
        top_k=1,
    )

    assert results
    assert results[0]["metadata"]["source"] == "refund_policy.txt"
    assert results[0]["metadata"]["document_type"] == "policy"
    assert results[0]["metadata"]["chunk_id"] == "refund_policy.txt:0"



def test_semantic_retriever_passes_metadata_filter():
    class FilterAwareRetriever:
        def __init__(self):
            self.received_where = None

        def retrieve(self, query_embedding, where=None):
            self.received_where = where

            return {
                "ids": [["policy1"]],
                "documents": [["Refund policy information."]],
                "distances": [[0.1]],
                "metadatas": [[
                    {
                        "document_type": "policy",
                    }
                ]],
            }

    class FakeEmbeddingModel:
        def embed_documents(self, documents):
            return [[0.1, 0.2, 0.3] for _ in documents]

    retriever = FilterAwareRetriever()
    embedding_model = FakeEmbeddingModel()

    adapter = SemanticRetrieverAdapter(
        retriever,
        embedding_model,
    )

    results = adapter.search(
        "refund",
        top_k=1,
        where={"document_type": "policy"},
    )

    assert results
    assert retriever.received_where == {
        "document_type": "policy"
    }
