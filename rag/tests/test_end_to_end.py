from rag.langgraph_workflow import LangGraphAgent


def test_end_to_end_search_requires_human_approval():
    calls = []

    def search_documents(query):
        calls.append(query)

        return [
            {
                "id": "refund_policy.md:0",
                "text": "Customers can request a refund within 30 days.",
                "metadata": {
                    "source": "refund_policy.md",
                    "document_type": "policy",
                },
            }
        ]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    result = agent.run(
        "search_documents",
        query="What is the refund policy?"
    )

    assert result["status"] == "pending"
    assert calls == []


def test_end_to_end_approved_search_returns_results():
    calls = []

    def search_documents(query):
        calls.append(query)

        return [
            {
                "id": "refund_policy.md:0",
                "text": "Customers can request a refund within 30 days.",
                "metadata": {
                    "source": "refund_policy.md",
                    "document_type": "policy",
                },
            }
        ]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    agent.approve()

    result = agent.run(
        "search_documents",
        query="What is the refund policy?"
    )

    assert result["status"] == "approved"
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "refund_policy.md:0"
    assert "30 days" in result["results"][0]["text"]
    assert calls == ["What is the refund policy?"]


def test_end_to_end_rejected_search_returns_no_results():
    calls = []

    def search_documents(query):
        calls.append(query)

        return [
            {
                "id": "refund_policy.md:0",
                "text": "Customers can request a refund within 30 days.",
            }
        ]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    agent.reject()

    result = agent.run(
        "search_documents",
        query="What is the refund policy?"
    )

    assert result["status"] == "rejected"
    assert "results" not in result
    assert calls == []


def test_end_to_end_query_is_normalized_before_search():
    calls = []

    def search_documents(query):
        calls.append(query)
        return []

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    agent.approve()

    agent.run(
        "search_documents",
        query="   refund policy   "
    )

    assert calls == ["refund policy"]