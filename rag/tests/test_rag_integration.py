from rag.rag import RAGPipeline
from rag.llm import FakeLLM


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRetriever:
    def retrieve(self, query_embedding):
        return {
            "documents": [
                [
                    "Customers can request a refund within 30 days."
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "refund_policy.md",
                        "chunk_id": "refund_policy.md:0",
                        "chunk_index": 0,
                        "total_chunks": 1,
                    }
                ]
            ],
        }


def test_rag_pipeline_end_to_end():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    answer = pipeline.answer(
        "Can customers request a refund?"
    )

    assert "30 days" in answer