import pytest

from rag.prompt import build_prompt


def test_build_prompt_contains_question():
    prompt = build_prompt(
        "What is the refund period?",
        "Refunds are available within 30 days.",
    )

    assert "What is the refund period?" in prompt


def test_build_prompt_contains_context():
    prompt = build_prompt(
        "What is the refund period?",
        "Refunds are available within 30 days.",
    )

    assert "Refunds are available within 30 days." in prompt


def test_build_prompt_contains_instruction():
    prompt = build_prompt(
        "What is the refund period?",
        "Refunds are available within 30 days.",
    )

    assert "Do not make up information." in prompt


def test_empty_question_raises_error():
    with pytest.raises(ValueError):
        build_prompt(
            "",
            "Refunds are available within 30 days.",
        )


def test_empty_context_raises_error():
    with pytest.raises(ValueError):
        build_prompt(
            "What is the refund period?",
            "",
        )