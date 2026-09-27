from rag.agent import ConstrainedAgent


def test_agent_starts_with_allowed_actions():
    agent = ConstrainedAgent()

    assert agent.allowed_actions == [
        "search_documents",
        "get_document",
        "list_documents",
    ]


def test_agent_rejects_unknown_action():
    agent = ConstrainedAgent()

    try:
        agent.execute("delete_database")
        assert False
    except ValueError:
        assert True


def test_agent_rejects_empty_action():
    agent = ConstrainedAgent()

    try:
        agent.execute("")
        assert False
    except ValueError:
        assert True


def test_agent_executes_search_documents():
    agent = ConstrainedAgent()

    result = agent.execute(
        "search_documents",
        query="refund policy",
    )

    assert result["action"] == "search_documents"
    assert result["query"] == "refund policy"


def test_agent_executes_get_document():
    agent = ConstrainedAgent()

    result = agent.execute(
        "get_document",
        document_id="order_policy.md:0",
    )

    assert result["action"] == "get_document"
    assert result["document_id"] == "order_policy.md:0"


def test_agent_calls_search_tool():
    calls = []

    def search_tool(query):
        calls.append(query)
        return ["order_policy.md:0"]

    agent = ConstrainedAgent(
        search_documents=search_tool,
    )

    result = agent.execute(
        "search_documents",
        query="refund policy",
    )

    assert calls == ["refund policy"]
    assert result["results"] == ["order_policy.md:0"]


def test_agent_calls_get_document_tool():
    calls = []

    def get_document_tool(document_id):
        calls.append(document_id)
        return "Customers can request a refund within 30 days."

    agent = ConstrainedAgent(
        get_document=get_document_tool,
    )

    result = agent.execute(
        "get_document",
        document_id="order_policy.md:0",
    )

    assert calls == ["order_policy.md:0"]
    assert result["document"] == (
        "Customers can request a refund within 30 days."
    )


def test_search_documents_requires_query():
    agent = ConstrainedAgent()

    try:
        agent.execute("search_documents")
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "query is required for search_documents"
        )


def test_get_document_requires_document_id():
    agent = ConstrainedAgent()

    try:
        agent.execute("get_document")
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "document_id is required for get_document"
        )


def test_search_tool_is_not_called_without_query():
    calls = []

    def search_tool(query):
        calls.append(query)
        return ["order_policy.md:0"]

    agent = ConstrainedAgent(
        search_documents=search_tool,
    )

    try:
        agent.execute("search_documents")
        assert False
    except ValueError:
        assert calls == []


def test_get_document_tool_is_not_called_without_document_id():
    calls = []

    def get_document_tool(document_id):
        calls.append(document_id)
        return "document"

    agent = ConstrainedAgent(
        get_document=get_document_tool,
    )

    try:
        agent.execute("get_document")
        assert False
    except ValueError:
        assert calls == []


def test_agent_integrates_with_rag_style_tools():
    documents = {
        "order_policy.md:0": (
            "Customers can request a refund within 30 days."
        ),
        "customer_support.md:0": (
            "Customers can check order status using the support portal."
        ),
    }

    def search_documents(query):
        query_lower = query.lower()

        results = []

        for document_id, text in documents.items():
            if "refund" in query_lower and "refund" in text.lower():
                results.append(document_id)

            if "order status" in query_lower and "order status" in text.lower():
                results.append(document_id)

        return results

    def get_document(document_id):
        return documents.get(document_id)

    agent = ConstrainedAgent(
        search_documents=search_documents,
        get_document=get_document,
    )

    search_result = agent.execute(
        "search_documents",
        query="refund policy",
    )

    assert search_result["results"] == [
        "order_policy.md:0"
    ]

    document_result = agent.execute(
        "get_document",
        document_id="order_policy.md:0",
    )

    assert document_result["document"] == (
        "Customers can request a refund within 30 days."
    )

def test_agent_allows_list_documents():
    agent = ConstrainedAgent()

    assert agent.allowed_actions == [
        "search_documents",
        "get_document",
        "list_documents",
    ]


def test_agent_calls_list_documents_tool():
    calls = []

    def list_documents_tool():
        calls.append("called")
        return [
            "order_policy.md",
            "customer_support.md",
            "churn_strategy.md",
        ]

    agent = ConstrainedAgent(
        list_documents=list_documents_tool,
    )

    result = agent.execute(
        "list_documents",
    )

    assert calls == ["called"]
    assert result["documents"] == [
        "order_policy.md",
        "customer_support.md",
        "churn_strategy.md",
    ]


def test_list_documents_does_not_accept_unexpected_arguments():
    def list_documents_tool():
        return ["order_policy.md"]

    agent = ConstrainedAgent(
        list_documents=list_documents_tool,
    )

    try:
        agent.execute(
            "list_documents",
            query="refund",
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "list_documents does not accept arguments"
        )


def test_agent_records_search_trace(tmp_path):
    from rag.agent_trace import AgentTraceStore

    trace_file = tmp_path / "agent_traces.json"
    trace_store = AgentTraceStore(trace_file)

    def search_documents_tool(query):
        return [
            {
                "id": "order_policy.md:0",
                "text": "Refunds are allowed.",
            }
        ]

    agent = ConstrainedAgent(
        search_documents=search_documents_tool,
        trace_store=trace_store,
    )

    result = agent.execute(
        "search_documents",
        query="refund policy",
    )

    assert result["results"] == [
        {
            "id": "order_policy.md:0",
            "text": "Refunds are allowed.",
        }
    ]

    traces = trace_store.get_traces()

    assert len(traces) == 1
    assert traces[0]["action"] == "search_documents"
    assert traces[0]["arguments"] == {
        "query": "refund policy"
    }
    assert traces[0]["result"] == result


def test_agent_records_get_document_trace(tmp_path):
    from rag.agent_trace import AgentTraceStore

    trace_file = tmp_path / "agent_traces.json"
    trace_store = AgentTraceStore(trace_file)

    def get_document_tool(document_id):
        return {
            "id": document_id,
            "text": "Refunds are allowed.",
        }

    agent = ConstrainedAgent(
        get_document=get_document_tool,
        trace_store=trace_store,
    )

    result = agent.execute(
        "get_document",
        document_id="order_policy.md:0",
    )

    traces = trace_store.get_traces()

    assert len(traces) == 1
    assert traces[0]["action"] == "get_document"
    assert traces[0]["arguments"] == {
        "document_id": "order_policy.md:0"
    }
    assert traces[0]["result"] == result


def test_agent_records_list_documents_trace(tmp_path):
    from rag.agent_trace import AgentTraceStore

    trace_file = tmp_path / "agent_traces.json"
    trace_store = AgentTraceStore(trace_file)

    def list_documents_tool():
        return [
            "order_policy.md",
            "customer_support.md",
        ]

    agent = ConstrainedAgent(
        list_documents=list_documents_tool,
        trace_store=trace_store,
    )

    result = agent.execute(
        "list_documents",
    )

    traces = trace_store.get_traces()

    assert len(traces) == 1
    assert traces[0]["action"] == "list_documents"
    assert traces[0]["arguments"] == {}
    assert traces[0]["result"] == result


def test_agent_records_token_usage_in_trace(tmp_path):
    from rag.agent_trace import AgentTraceStore

    trace_file = tmp_path / "agent_traces.json"
    trace_store = AgentTraceStore(trace_file)

    def search_documents_tool(query):
        return ["order_policy.md:0"]

    agent = ConstrainedAgent(
        search_documents=search_documents_tool,
        trace_store=trace_store,
    )

    agent.execute(
        "search_documents",
        query="refund policy",
        token_usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )

    traces = trace_store.get_traces()

    assert traces[0]["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }