import json

from rag.agent_trace import AgentTraceStore


def test_trace_store_starts_empty(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    store = AgentTraceStore(trace_file)

    assert store.get_traces() == []


def test_trace_store_records_agent_action(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    store = AgentTraceStore(trace_file)

    store.record(
        action="search_documents",
        arguments={"query": "refund policy"},
        result={"documents": ["order_policy.md:0"]},
    )

    traces = store.get_traces()

    assert len(traces) == 1
    assert traces[0]["action"] == "search_documents"
    assert traces[0]["arguments"] == {
        "query": "refund policy"
    }
    assert traces[0]["result"] == {
        "documents": ["order_policy.md:0"]
    }


def test_trace_store_persists_traces_to_file(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    store = AgentTraceStore(trace_file)

    store.record(
        action="list_documents",
        arguments={},
        result={
            "documents": [
                "order_policy.md",
                "customer_support.md",
            ]
        },
    )

    assert trace_file.exists()

    with open(trace_file, "r", encoding="utf-8") as file:
        saved_traces = json.load(file)

    assert saved_traces == [
        {
            "action": "list_documents",
            "arguments": {},
            "result": {
                "documents": [
                    "order_policy.md",
                    "customer_support.md",
                ]
            },
        }
    ]


def test_trace_store_loads_existing_traces(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    trace_file.write_text(
        json.dumps(
            [
                {
                    "action": "get_document",
                    "arguments": {
                        "document_id": "order_policy.md"
                    },
                    "result": {
                        "document": "Refunds are allowed."
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    store = AgentTraceStore(trace_file)

    assert store.get_traces() == [
        {
            "action": "get_document",
            "arguments": {
                "document_id": "order_policy.md"
            },
            "result": {
                "document": "Refunds are allowed."
            },
        }
    ]


def test_trace_store_appends_new_trace_to_existing_file(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    store = AgentTraceStore(trace_file)

    store.record(
        action="search_documents",
        arguments={"query": "refund"},
        result={"documents": ["order_policy.md:0"]},
    )

    store.record(
        action="list_documents",
        arguments={},
        result={"documents": ["order_policy.md"]},
    )

    assert store.get_traces() == [
        {
            "action": "search_documents",
            "arguments": {"query": "refund"},
            "result": {
                "documents": ["order_policy.md:0"]
            },
        },
        {
            "action": "list_documents",
            "arguments": {},
            "result": {
                "documents": ["order_policy.md"]
            },
        },
    ]

def test_trace_store_records_token_usage(tmp_path):
    trace_file = tmp_path / "agent_traces.json"

    store = AgentTraceStore(trace_file)

    store.record(
        action="search_documents",
        arguments={"query": "refund policy"},
        result={"documents": ["order_policy.md:0"]},
        token_usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )

    traces = store.get_traces()

    assert traces[0]["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }