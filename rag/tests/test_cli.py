from rag.cli import format_citations


def test_format_citations_includes_source_chunk_and_text():
    citations = [
        {
            "source": "refund_policy.md",
            "chunk_id": "refund_policy.md:0",
            "text": "Customers can request a refund within 30 days.",
        }
    ]

    output = format_citations(citations)

    assert "[1] refund_policy.md" in output
    assert "Chunk: refund_policy.md:0" in output
    assert '"Customers can request a refund within 30 days."' in output