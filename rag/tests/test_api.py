
from fastapi.testclient import TestClient

from rag.api import create_app


class FakeRAGPipeline:
    def answer_with_citations(
        self,
        question,
        where=None,
        use_recency=False,
    ):
        return {
            "answer": f"Answer for: {question}",
            "citations": [
                {
                    "source": "refund_policy.txt",
                    "chunk_id": "refund_policy.txt:chunk_0",
                    "text": "Refunds are allowed within 30 days.",
                }
            ],
        }


def create_test_client():
    pipeline = FakeRAGPipeline()
    app = create_app(rag_pipeline=pipeline)
    return TestClient(app)


def test_health_endpoint():
    client = create_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_query_endpoint():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "What is the refund policy?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == (
        "Answer for: What is the refund policy?"
    )

    assert len(body["citations"]) == 1
    assert body["citations"][0]["source"] == (
        "refund_policy.txt"
    )


def test_query_rejects_empty_question():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": ""
        },
    )

    assert response.status_code == 400


def test_query_rejects_missing_question():
    client = create_test_client()

    response = client.post(
        "/query",
        json={}
    )

    assert response.status_code == 422


def test_metrics_endpoint():
    client = create_test_client()

    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()

    assert "queries" in body
    assert "status" in body
    assert body["status"] == "ok"


def test_create_app_with_rag_pipeline():
    from rag.llm import FakeLLM
    from rag.rag import RAGPipeline

    class FakeStore:
        pass

    pipeline = RAGPipeline(
        FakeStore(),
        FakeLLM(),
        top_k=2,
    )

    app = create_app(
        rag_pipeline=pipeline
    )

    assert app is not None
    assert app.title == "RAG + Agent API"


def test_importing_api_does_not_build_rag_pipeline(monkeypatch):
    import importlib
    import sys

    if "rag.api" in sys.modules:
        del sys.modules["rag.api"]

    import rag.ingest

    def fail_if_called():
        raise AssertionError(
            "RAG pipeline should not be built during API import"
        )

    monkeypatch.setattr(
        rag.ingest,
        "ingest_documents",
        fail_if_called,
    )

    importlib.import_module("rag.api")



def test_api_exposes_fastapi_app():
    from fastapi import FastAPI
    from rag.api import app

    assert isinstance(app, FastAPI)


def test_app_builds_rag_pipeline_on_startup(monkeypatch):
    from fastapi.testclient import TestClient
    import rag.api

    class FakePipeline:
        def answer_with_citations(self, question, where=None, use_recency=False):
            return {
                "answer": "Startup pipeline works",
                "citations": [],
            }

    calls = []

    def fake_build_rag_pipeline():
        calls.append(1)
        return FakePipeline()

    monkeypatch.setattr(
        rag.api,
        "build_rag_pipeline",
        fake_build_rag_pipeline,
    )

    app = rag.api.create_app(
        pipeline_factory=rag.api.build_rag_pipeline
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"question": "Test startup"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Startup pipeline works"
    assert len(calls) == 1






