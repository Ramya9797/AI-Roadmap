from rag.tracing import TraceRecorder
from rag.rag import RAGPipeline
from rag.version import APP_VERSION, MODEL_VERSION


class FakeLLM:
    def generate(self, prompt):
        return "The refund policy allows refunds within 30 days."


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3]]


class FakeRetriever:
    def retrieve(self, query_embedding, where=None):
        return {
            "documents": [
                [
                    "Customers can request a refund within 30 days."
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "order_policy.md",
                        "chunk_id": "order_policy.md:0",
                    }
                ]
            ],
        }


def test_trace_recorder_starts_with_version_metadata():
    recorder = TraceRecorder()

    trace = recorder.get_trace()

    assert trace["app_version"] == "1.0.0"
    assert trace["model_version"] == "all-MiniLM-L6-v2"
    assert trace["prompt_version"] == "1.0.0"
    assert trace["index_version"] == "1.0.0"


def test_trace_recorder_records_question():
    recorder = TraceRecorder()

    recorder.record("question", "What is the refund policy?")

    trace = recorder.get_trace()

    assert trace["question"] == "What is the refund policy?"


def test_trace_recorder_records_rewritten_query():
    recorder = TraceRecorder()

    recorder.record("rewritten_query", "refund policy")

    trace = recorder.get_trace()

    assert trace["rewritten_query"] == "refund policy"


def test_trace_recorder_records_retrieved_chunks():
    recorder = TraceRecorder()

    recorder.record(
        "retrieved_chunk_ids",
        ["order_policy.md:0"],
    )

    trace = recorder.get_trace()

    assert trace["retrieved_chunk_ids"] == [
        "order_policy.md:0"
    ]


def test_rag_pipeline_records_trace():
    recorder = TraceRecorder()

    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()
    pipeline.query_rewriter = None
    pipeline.trace_recorder = recorder

    result = pipeline.answer_with_citations(
        "What is the refund policy?"
    )

    trace = recorder.get_trace()

    assert result["answer"] == (
        "The refund policy allows refunds within 30 days."
    )
    assert trace["question"] == "What is the refund policy?"
    assert trace["retrieved_chunk_ids"] == [
        "order_policy.md:0"
    ]
    assert trace["retrieval_count"] == 1


def test_trace_recorder_includes_version_metadata():
    recorder = TraceRecorder()

    recorder.record(
        "app_version",
        APP_VERSION,
    )
    recorder.record(
        "model_version",
        MODEL_VERSION,
    )

    trace = recorder.get_trace()

    assert trace["app_version"] == "1.0.0"
    assert trace["model_version"] == "all-MiniLM-L6-v2"