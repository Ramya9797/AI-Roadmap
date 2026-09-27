from rag.langgraph_workflow import LangGraphAgent


def test_complete_workflow_approval_then_search():
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

    # Before approval, execution must not happen.
    pending = agent.run(
        "search_documents",
        query="What is the refund policy?"
    )

    assert pending["status"] == "pending"
    assert calls == []

    # Human approves the action.
    agent.approve()

    approved = agent.run(
        "search_documents",
        query="What is the refund policy?"
    )

    assert approved["status"] == "approved"
    assert len(approved["results"]) == 1
    assert approved["results"][0]["id"] == "refund_policy.md:0"
    assert "30 days" in approved["results"][0]["text"]

    assert calls == [
        "What is the refund policy?"
    ]


def test_complete_workflow_rejection_stops_execution():
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