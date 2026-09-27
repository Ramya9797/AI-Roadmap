from rag.answer_evaluation import evaluate_answer


def test_answer_is_supported_by_context():
    result = evaluate_answer(
        answer="The refund policy allows refunds within 30 days.",
        context="Customers can request a refund within 30 days of purchase.",
    )

    assert result["supported"] is True


def test_answer_is_not_supported_by_context():
    result = evaluate_answer(
        answer="Customers can request refunds within 90 days.",
        context="Customers can request a refund within 30 days of purchase.",
    )

    assert result["supported"] is False


def test_answer_with_empty_context_is_not_supported():
    result = evaluate_answer(
        answer="The refund policy allows refunds.",
        context="",
    )

    assert result["supported"] is False


def test_answer_evaluation_rejects_empty_answer():
    try:
        evaluate_answer(
            answer="",
            context="Some refund policy information.",
        )
        assert False
    except ValueError:
        assert True


def test_answer_evaluation_with_empty_context_is_not_supported():
    result = evaluate_answer(
        answer="The refund policy allows refunds.",
        context="",
    )

    assert result["supported"] is False