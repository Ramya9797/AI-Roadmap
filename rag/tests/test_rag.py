from rag.rag import RAGPipeline


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "Refunds are available within 30 days."


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRetriever:
    def retrieve(self, query_embedding):
        return {
            "documents": [
                [
                    "Refunds are available within 30 days."
                ]
            ]
        }


def test_rag_returns_answer():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    answer = pipeline.answer(
        "What is the refund period?"
    )

    assert answer == "Refunds are available within 30 days."


def test_rag_rejects_empty_question():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    try:
        pipeline.answer("")
        assert False
    except ValueError:
        assert True