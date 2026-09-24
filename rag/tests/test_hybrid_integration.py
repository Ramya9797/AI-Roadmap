from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.metadata import add_metadata
from rag.keyword_retriever import KeywordRetriever
from rag.embeddings import EmbeddingModel
from rag.semantic_retriever import SemanticRetrieverAdapter
from rag.vector_store import ChromaVectorStore
from rag.retriever import Retriever
from rag.hybrid_search import HybridRetriever


def test_keyword_retriever_works_with_real_documents():
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)

    keyword_documents = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    retriever = KeywordRetriever(keyword_documents)

    results = retriever.search("refund")

    assert results
    assert results[0]["score"] > 0


def test_keyword_retriever_finds_order_policy():
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)

    keyword_documents = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    retriever = KeywordRetriever(keyword_documents)

    results = retriever.search("order")

    assert results
    assert any(
        "order" in result["text"].lower()
        for result in results
    )


def test_keyword_retriever_finds_rare_term():
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)

    keyword_documents = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    retriever = KeywordRetriever(keyword_documents)

    results = retriever.search("proactive")

    assert results
    assert results[0]["score"] > 0




def test_hybrid_retriever_works_with_real_documents():
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)

    # Keyword retrieval documents
    keyword_documents = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    keyword_retriever = KeywordRetriever(keyword_documents)

    # Create a fresh Chroma collection for this test
    store = ChromaVectorStore(
        collection_name="hybrid_integration_test"
    )

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks,
        embeddings,
    )

    semantic_retriever = SemanticRetrieverAdapter(
        Retriever(store, top_k=3),
        embedding_model,
    )

    hybrid_retriever = HybridRetriever(
        semantic_retriever,
        keyword_retriever,
    )

    results = hybrid_retriever.search(
        "refund",
        top_k=3,
    )

    assert results
    assert len(results) <= 3

    for result in results:
        assert "id" in result
        assert "text" in result
        assert "semantic_score" in result
        assert "keyword_score" in result
        assert "combined_score" in result


def test_hybrid_retriever_supports_metadata_filter():
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)

    keyword_documents = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": {
                "source": chunk["source"],
                "document_type": chunk["document_type"],
                "document_name": chunk["document_name"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
            },
        }
        for chunk in chunks
    ]

    keyword_retriever = KeywordRetriever(keyword_documents)

    store = ChromaVectorStore(
        collection_name="hybrid_metadata_filter_test"
    )

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(chunks, embeddings)

    semantic_retriever = SemanticRetrieverAdapter(
        Retriever(store, top_k=3),
        embedding_model,
    )

    hybrid = HybridRetriever(
        semantic_retriever,
        keyword_retriever,
    )

    results = hybrid.search(
        "refund",
        top_k=3,
        where={"document_type": "policy"},
    )

    assert results

    assert all(
        result["metadata"]["document_type"] == "policy"
        for result in results
    )