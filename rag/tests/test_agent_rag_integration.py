from rag.agent import ConstrainedAgent


def test_agent_search_uses_real_rag_style_retriever():
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

    agent = ConstrainedAgent(
        search_documents=search_documents,
    )

    result = agent.execute(
        "search_documents",
        query="What is the refund policy?",
    )

    assert calls == ["What is the refund policy?"]

    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "refund_policy.md:0"
    assert "30 days" in result["results"][0]["text"]