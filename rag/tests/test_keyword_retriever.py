from rag.keyword_retriever import KeywordRetriever


DOCUMENTS = [
    {
        "id": "doc-1",
        "text": "Customers can request a refund within 30 days.",
    },
    {
        "id": "doc-2",
        "text": "Order ID ORD-12345 has been shipped.",
    },
    {
        "id": "doc-3",
        "text": "Customers can contact support for technical issues.",
    },
]


def test_keyword_search_returns_results():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search("refund")

    assert results


def test_keyword_search_matches_exact_keyword():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search("refund")

    assert results[0]["id"] == "doc-1"


def test_keyword_search_matches_order_id():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search("ORD-12345")

    assert results[0]["id"] == "doc-2"


def test_keyword_search_matches_rare_term():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search("technical")

    assert results[0]["id"] == "doc-3"


def test_keyword_search_supports_top_k():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search(
        "customers",
        top_k=2,
    )

    assert len(results) == 2


def test_keyword_search_returns_keyword_score():
    retriever = KeywordRetriever(DOCUMENTS)

    results = retriever.search("refund")

    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_keyword_search_empty_query():
    retriever = KeywordRetriever(DOCUMENTS)

    try:
        retriever.search("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_keyword_retriever_preserves_metadata():
    documents = [
        {
            "id": "doc1",
            "text": "Refund policy information.",
            "metadata": {
                "source": "refund_policy.txt",
                "document_type": "policy",
                "document_name": "refund_policy",
                "chunk_index": 0,
                "total_chunks": 2,
            },
        }
    ]

    retriever = KeywordRetriever(documents)

    results = retriever.search("refund")

    assert results
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"]["source"] == "refund_policy.txt"
    assert results[0]["metadata"]["document_type"] == "policy"


def test_keyword_retriever_supports_metadata_filter():
    documents = [
        {
            "id": "policy1",
            "text": "Refund policy information.",
            "metadata": {
                "document_type": "policy",
            },
        },
        {
            "id": "support1",
            "text": "Refund support information.",
            "metadata": {
                "document_type": "support",
            },
        },
    ]

    retriever = KeywordRetriever(documents)

    results = retriever.search(
        "refund",
        top_k=3,
        where={"document_type": "policy"},
    )

    assert results
    assert all(
        result["metadata"]["document_type"] == "policy"
        for result in results
    )
    assert all(
        result["id"] == "policy1"
        for result in results
    )