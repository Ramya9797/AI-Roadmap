from rag.langgraph_workflow import LangGraphAgent


def test_langgraph_agent_can_execute_search():
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

    assert calls == [
        "What is the refund policy?"
    ]

    assert result["results"][0]["id"] == "refund_policy.md:0"

    assert "30 days" in result["results"][0]["text"]


def test_langgraph_agent_rejects_unknown_action():
    agent = LangGraphAgent(
        search_documents=lambda query: []
    )

    try:
        agent.run("unknown_action")
        assert False
    except ValueError:
        assert True


def test_langgraph_agent_preserves_search_results():
    def search_documents(query):
        return [
            {
                "id": "order_policy.md:0",
                "text": "Orders can be cancelled within 24 hours.",
                "metadata": {
                    "source": "order_policy.md",
                    "document_type": "policy",
                },
            },
            {
                "id": "support.md:0",
                "text": "Contact support for order cancellation assistance.",
                "metadata": {
                    "source": "support.md",
                    "document_type": "support",
                },
            },
        ]

    agent = LangGraphAgent(
        search_documents=search_documents
    )
    agent.approve()

    result = agent.run(
        "search_documents",
        query="How can I cancel my order?"
    )

    assert len(result["results"]) == 2
    assert result["results"][0]["id"] == "order_policy.md:0"
    assert result["results"][1]["id"] == "support.md:0"


def test_langgraph_agent_normalizes_query():
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


def test_langgraph_agent_rejects_empty_query():
    agent = LangGraphAgent(
        search_documents=lambda query: []
    )

    try:
        agent.run(
            "search_documents",
            query="   "
        )
        assert False
    except ValueError as exc:
        assert str(exc) == "query is required"


def test_langgraph_requires_approval_before_execution():
    calls = []

    def search_documents(query):
        calls.append(query)
        return [{"id": "doc1"}]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    result = agent.run(
        "search_documents",
        query="refund policy",
    )

    assert result["status"] == "pending"
    assert calls == []


def test_langgraph_executes_after_approval():
    calls = []

    def search_documents(query):
        calls.append(query)
        return [{"id": "doc1"}]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    agent.approve()

    result = agent.run(
        "search_documents",
        query="refund policy",
    )

    assert result["status"] == "approved"
    assert result["results"] == [{"id": "doc1"}]
    assert calls == ["refund policy"]


def test_langgraph_does_not_execute_after_rejection():
    calls = []

    def search_documents(query):
        calls.append(query)
        return [{"id": "doc1"}]

    agent = LangGraphAgent(
        search_documents=search_documents
    )

    agent.reject()

    result = agent.run(
        "search_documents",
        query="refund policy",
    )

    assert result["status"] == "rejected"
    assert calls == []