from rag.rag import RAGPipeline


class FakeLLM:
    def generate(self, prompt):
        return "Customers can request a refund within 30 days."


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query_embedding, where=None):
        return self.results


class FakeEmbeddingModel:
    def embed_documents(self, documents):
        return [[0.1, 0.2, 0.3] for _ in documents]


def create_pipeline(results):
    pipeline = RAGPipeline(
        store=None,
        llm=FakeLLM(),
    )

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever(results)

    return pipeline


def test_final_rag_pipeline_returns_answer_and_citation():
    results = {
        "documents": [[
            "Customers can request a refund within 30 days."
        ]],
        "metadatas": [[
            {
                "source": "refund_policy.md",
                "chunk_id": "refund_policy.md:0",
            }
        ]],
    }

    pipeline = create_pipeline(results)

    result = pipeline.answer_with_citations(
        "What is the refund policy?"
    )

    assert "30 days" in result["answer"]

    assert len(result["citations"]) == 1

    assert (
        result["citations"][0]["chunk_id"]
        == "refund_policy.md:0"
    )

    assert (
        result["citations"][0]["source"]
        == "refund_policy.md"
    )


def test_final_rag_pipeline_refuses_empty_retrieval():
    results = {
        "documents": [[]],
        "metadatas": [[]],
    }

    pipeline = create_pipeline(results)

    result = pipeline.answer_with_citations(
        "What is something unrelated?"
    )

    assert (
        result["answer"]
        == "I don't have enough information to answer that."
    )

    assert result["citations"] == []