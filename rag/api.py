from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.ingest import ingest_documents
from rag.llm import FakeLLM
from rag.rag import RAGPipeline


class QueryRequest(BaseModel):
    question: str


def build_rag_pipeline():
    """
    Build the real RAG pipeline.

    This is called during application startup,
    not during module import.
    """
    store = ingest_documents()

    llm = FakeLLM()

    return RAGPipeline(
        store,
        llm,
        top_k=2,
    )


def create_app(rag_pipeline=None, pipeline_factory=None):
    pipeline = rag_pipeline

    @asynccontextmanager
    async def lifespan(app):
        nonlocal pipeline

        if pipeline is None and pipeline_factory is not None:
            pipeline = pipeline_factory()

        yield

    app = FastAPI(
        title="RAG + Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    metrics = {
        "queries": 0,
    }

    @app.get("/health")
    def health():
        return {
            "status": "ok"
        }

    @app.post("/query")
    def query(request: QueryRequest):
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty",
            )

        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline is not configured",
            )

        metrics["queries"] += 1

        result = pipeline.answer_with_citations(
            question
        )

        return {
            "answer": result["answer"],
            "citations": result["citations"],
        }

    @app.get("/metrics")
    def get_metrics():
        return {
            "status": "ok",
            "queries": metrics["queries"],
        }

    return app


app = create_app(
    pipeline_factory=build_rag_pipeline
)