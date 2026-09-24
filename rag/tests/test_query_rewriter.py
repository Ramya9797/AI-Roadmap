from rag.query_rewriter import QueryRewriter


def test_query_rewriter_expands_vague_query():
    rewriter = QueryRewriter()

    rewritten = rewriter.rewrite(
        "What is the refund thing?"
    )

    assert rewritten == "refund policy"


def test_query_rewriter_keeps_specific_query():
    rewriter = QueryRewriter()

    rewritten = rewriter.rewrite(
        "What is the refund policy?"
    )

    assert rewritten == "What is the refund policy?"


def test_query_rewriter_rejects_empty_query():
    rewriter = QueryRewriter()

    try:
        rewriter.rewrite("")
        assert False
    except ValueError:
        assert True


def test_query_rewriter_handles_vague_refund_question():
    rewriter = QueryRewriter()

    rewritten = rewriter.rewrite(
        "Tell me about refunds"
    )

    assert rewritten == "refund policy"


def test_query_rewriter_handles_vague_order_question():
    rewriter = QueryRewriter()

    rewritten = rewriter.rewrite(
        "What about my order?"
    )

    assert rewritten == "order status"

def test_query_rewriter_uses_rewriter_model():
    class FakeRewriterModel:
        def rewrite(self, query):
            return "refund policy"

    rewriter = QueryRewriter(
        model=FakeRewriterModel()
    )

    rewritten = rewriter.rewrite(
        "Can you explain how refunds work?"
    )

    assert rewritten == "refund policy"


def test_query_rewriter_rejects_empty_model_output():
    class FakeRewriterModel:
        def rewrite(self, query):
            return ""

    rewriter = QueryRewriter(
        model=FakeRewriterModel()
    )

    try:
        rewriter.rewrite(
            "Can you explain refunds?"
        )
        assert False
    except ValueError:
        assert True


def test_query_rewriter_passes_normalized_query_to_model():
    class FakeRewriterModel:
        def __init__(self):
            self.received_query = None

        def rewrite(self, query):
            self.received_query = query
            return "refund policy"

    model = FakeRewriterModel()

    rewriter = QueryRewriter(
        model=model
    )

    rewritten = rewriter.rewrite(
        "   Tell me about refunds   "
    )

    assert rewritten == "refund policy"
    assert model.received_query == "Tell me about refunds"