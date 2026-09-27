from datetime import datetime, timedelta

from rag.rag import RAGPipeline
from rag.recency import calculate_recency_score


class FakeLLM:
    def generate(self, prompt):
        return "Recent refund policy information."


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRetriever:
    def __init__(self):
        self.use_recency = None

    def retrieve(
        self,
        query_embedding,
        where=None,
        use_recency=False,
    ):
        self.use_recency = use_recency

        now = datetime.now()

        results = [
            {
                "document": "Old refund policy",
                "metadata": {
                    "source": "old.md",
                    "chunk_id": "old:0",
                    "created_at": (
                        now - timedelta(days=30)
                    ).isoformat(),
                },
            },
            {
                "document": "Recent refund policy",
                "metadata": {
                    "source": "recent.md",
                    "chunk_id": "recent:0",
                    "created_at": (
                        now - timedelta(days=1)
                    ).isoformat(),
                },
            },
        ]

        if use_recency:
            for item in results:
                created_at = datetime.fromisoformat(
                    item["metadata"]["created_at"]
                )

                item["score"] = calculate_recency_score(
                    created_at=created_at,
                    now=now,
                )

            results.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

        return {
            "documents": [[
                item["document"]
                for item in results
            ]],
            "metadatas": [[
                item["metadata"]
                for item in results
            ]],
        }


def test_rag_can_enable_recency():
    retriever = FakeRetriever()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = retriever
    pipeline.llm = FakeLLM()

    result = pipeline.answer_with_citations(
        "What is the refund policy?",
        use_recency=True,
    )

    assert result["citations"][0]["chunk_id"] == "recent:0"
    assert retriever.use_recency is True