from rag.hybrid_search import HybridRetriever


class FakeSemanticRetriever:
    def search(self, query, top_k=3, where=None):
        return [
            {
                "id": "semantic-1",
                "text": "Orders can be refunded within the refund period.",
                "score": 0.90,
            },
            {
                "id": "semantic-2",
                "text": "Customers can contact support for order issues.",
                "score": 0.80,
            },
        ][:top_k]


class FakeKeywordRetriever:
    def search(self, query, top_k=3, where=None):
        return [
            {
                "id": "keyword-1",
                "text": "Order ID ORD-12345 has been refunded.",
                "score": 1.00,
            },
            {
                "id": "keyword-2",
                "text": "Order IDs are used to identify customer orders.",
                "score": 0.70,
            },
        ][:top_k]


def test_hybrid_search_returns_results():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search("What is the refund policy?")

    assert results
    assert len(results) > 0


def test_hybrid_search_combines_semantic_and_keyword_results():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search("refund")

    result_ids = [result["id"] for result in results]

    assert "semantic-1" in result_ids
    assert "keyword-1" in result_ids


def test_hybrid_search_exact_order_id():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search("What is order ID ORD-12345?")

    assert results[0]["id"] == "keyword-1"


def test_hybrid_search_prefers_exact_keyword_match():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search("ORD-12345")

    assert results[0]["id"] == "keyword-1"


def test_hybrid_search_supports_top_k():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search(
        "refund",
        top_k=2,
    )

    assert len(results) == 2



def test_hybrid_search_exact_identifier_gets_boost():
    semantic_retriever = FakeSemanticRetriever()
    keyword_retriever = FakeKeywordRetriever()

    retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        semantic_weight=0.5,
        keyword_weight=0.5,
        exact_match_boost=0.2,
    )

    results = retriever.search("ORD-12345")

    assert results[0]["id"] == "keyword-1"
    assert results[0]["exact_match"] is True


def test_hybrid_search_returns_scores():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    results = retriever.search("refund")

    assert "semantic_score" in results[0]
    assert "keyword_score" in results[0]
    assert "combined_score" in results[0]



def test_hybrid_search_rejects_empty_query():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    try:
        retriever.search("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_hybrid_search_rejects_invalid_top_k():
    retriever = HybridRetriever(
        semantic_retriever=FakeSemanticRetriever(),
        keyword_retriever=FakeKeywordRetriever(),
    )

    try:
        retriever.search("refund", top_k=0)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than 0"


def test_hybrid_search_rejects_negative_semantic_weight():
    try:
        HybridRetriever(
            semantic_retriever=FakeSemanticRetriever(),
            keyword_retriever=FakeKeywordRetriever(),
            semantic_weight=-0.1,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "semantic_weight cannot be negative"


def test_hybrid_search_rejects_negative_keyword_weight():
    try:
        HybridRetriever(
            semantic_retriever=FakeSemanticRetriever(),
            keyword_retriever=FakeKeywordRetriever(),
            keyword_weight=-0.1,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "keyword_weight cannot be negative"


def test_hybrid_search_rejects_zero_weights():
    try:
        HybridRetriever(
            semantic_retriever=FakeSemanticRetriever(),
            keyword_retriever=FakeKeywordRetriever(),
            semantic_weight=0.0,
            keyword_weight=0.0,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == (
            "At least one retrieval weight must be greater than 0"
        )


def test_hybrid_search_rejects_negative_exact_match_boost():
    try:
        HybridRetriever(
            semantic_retriever=FakeSemanticRetriever(),
            keyword_retriever=FakeKeywordRetriever(),
            exact_match_boost=-0.1,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "exact_match_boost cannot be negative"


def test_hybrid_search_preserves_metadata():
    class FakeRetriever:
        def __init__(self, results):
            self.results = results

        def search(self, query, top_k=3, where=None):
            return self.results[:top_k]

    metadata = {
        "source": "refund_policy.txt",
        "document_type": "policy",
        "document_name": "refund_policy",
        "chunk_id": "refund_policy.txt:0",
        "chunk_index": 0,
        "total_chunks": 2,
    }

    semantic_retriever = FakeRetriever([
        {
            "id": "doc1",
            "text": "Refunds are allowed.",
            "score": 0.9,
            "metadata": metadata,
        }
    ])

    keyword_retriever = FakeRetriever([
        {
            "id": "doc1",
            "text": "Refunds are allowed.",
            "score": 2.0,
            "metadata": metadata,
        }
    ])

    retriever = HybridRetriever(
        semantic_retriever,
        keyword_retriever,
    )

    results = retriever.search(
        "refund",
        top_k=1,
    )

    assert results
    assert results[0]["metadata"]["source"] == "refund_policy.txt"
    assert results[0]["metadata"]["document_type"] == "policy"
    assert results[0]["metadata"]["chunk_id"] == "refund_policy.txt:0"


def test_hybrid_search_passes_metadata_filter_to_semantic_retriever():
    class FilterAwareRetriever:
        def __init__(self, results):
            self.results = results
            self.received_where = None

        def search(self, query, top_k=3, where=None):
            self.received_where = where
            return self.results[:top_k]

    semantic_retriever = FilterAwareRetriever([
        {
            "id": "policy1",
            "text": "Refund policy information.",
            "score": 0.9,
            "metadata": {
                "document_type": "policy",
            },
        }
    ])

    keyword_retriever = FilterAwareRetriever([
        {
            "id": "policy1",
            "text": "Refund policy information.",
            "score": 2.0,
            "metadata": {
                "document_type": "policy",
            },
        }
    ])

    retriever = HybridRetriever(
        semantic_retriever,
        keyword_retriever,
    )

    results = retriever.search(
        "refund",
        top_k=1,
        where={"document_type": "policy"},
    )

    assert results
    assert semantic_retriever.received_where == {
        "document_type": "policy"
    }