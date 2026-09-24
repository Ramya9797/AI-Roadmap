from rag.ingest import ingest_documents
from rag.rag import RAGPipeline
from rag.llm import FakeLLM


def test_rag_with_real_documents():
    store = ingest_documents()

    pipeline = RAGPipeline(
        store=store,
        llm=FakeLLM(),
        top_k=2,
    )

    answer = pipeline.answer(
        "Can customers request a refund?"
    )

    assert answer
    assert "30 days" in answer