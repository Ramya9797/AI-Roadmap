from rag.rag import RAGPipeline
from rag.llm import FakeLLM
from rag.vector_store import ChromaVectorStore
from rag.embeddings import EmbeddingModel

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


def test_rag_answer_with_citations():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    result = pipeline.answer_with_citations(
        "What is the refund period?"
    )

    assert "answer" in result
    assert "citations" in result

    assert result["answer"] == (
        "Refunds are available within 30 days."
    )

    assert len(result["citations"]) > 0

    citation = result["citations"][0]

    assert citation["source"] == "refund_policy.md"
    assert citation["chunk_id"] == "refund_policy.md:0"
    assert citation["text"] == (
        "Refunds are available within 30 days."
    )



def test_rag_citation_matches_retrieved_chunk():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()

    result = pipeline.answer_with_citations(
        "What is the refund period?"
    )

    citation = result["citations"][0]

    retrieved = pipeline.retriever.retrieve(
        [1.0, 0.0, 0.0]
    )

    retrieved_document = retrieved["documents"][0][0]
    retrieved_metadata = retrieved["metadatas"][0][0]

    assert citation["text"] == retrieved_document
    assert citation["source"] == retrieved_metadata["source"]
    assert citation["chunk_id"] == retrieved_metadata["chunk_id"]



def test_rag_refuses_when_retrieval_is_empty():
    class EmptyRetriever:
        def retrieve(self, query_embedding):
            return {
                "documents": [[]],
                "metadatas": [[]],
            }

    class FailingLLM:
        def generate(self, prompt):
            raise AssertionError(
                "LLM should not be called when retrieval is empty"
            )

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = EmptyRetriever()
    pipeline.llm = FailingLLM()

    result = pipeline.answer_with_citations(
        "What is the maternity leave policy?"
    )

    assert result["answer"] == (
        "I don't have enough information to answer that."
    )

    assert result["citations"] == []


def test_rag_calls_llm_when_retrieval_has_context():
    class TrackingLLM:
        def __init__(self):
            self.called = False

        def generate(self, prompt):
            self.called = True
            return "Refunds are available within 30 days."

    llm = TrackingLLM()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()
    pipeline.llm = llm

    result = pipeline.answer_with_citations(
        "What is the refund period?"
    )

    assert llm.called is True
    assert result["answer"] == (
        "Refunds are available within 30 days."
    )
    assert len(result["citations"]) > 0



def test_rag_supports_metadata_filter():
    store = ChromaVectorStore(
        collection_name="rag_metadata_filter_test_v2"
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

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )


    store.add_chunks(chunks, embeddings)

    llm = FakeLLM()
    pipeline = RAGPipeline(
        store,
        llm,
        top_k=5,
    )

    result = pipeline.answer_with_citations(
        "What is the refund policy?",
        where={"document_type": "policy"},
    )

    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "refund_policy.md"
    assert result["citations"][0]["chunk_id"] == "refund_policy.md:0"

def test_rag_pipeline_uses_query_rewriter():
    class FakeQueryRewriter:
        def __init__(self):
            self.received_query = None

        def rewrite(self, query):
            self.received_query = query
            return "refund policy"

    class FakeRetriever:
        def retrieve(self, query_embedding, where=None):
            return {
                "ids": [["policy:0"]],
                "documents": [["Refund policy information."]],
                "metadatas": [[
                    {
                        "source": "refund_policy.txt",
                        "chunk_id": "policy:0",
                    }
                ]],
                "distances": [[0.1]],
            }

    class FakeEmbeddingModel:
        def embed_documents(self, documents):
            self.documents = documents
            return [[0.1, 0.2, 0.3] for _ in documents]

    class FakeLLM:
        def generate(self, prompt):
            return "Refund information."

    rewriter = FakeQueryRewriter()

    pipeline = RAGPipeline(
        store=None,
        llm=FakeLLM(),
        top_k=2,
        query_rewriter=rewriter,
    )

    pipeline.embedding_model = FakeEmbeddingModel()
    pipeline.retriever = FakeRetriever()

    result = pipeline.answer_with_citations(
        "   Tell me about refunds   "
    )

    assert result["answer"] == "Refund information."
    assert rewriter.received_query == "Tell me about refunds"
    assert pipeline.embedding_model.documents == ["refund policy"]
