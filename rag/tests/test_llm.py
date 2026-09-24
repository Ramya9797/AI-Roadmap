# import pytest

# from rag.llm import LLM


# def test_llm_requires_api_key(monkeypatch):
#     monkeypatch.delenv("OPENAI_API_KEY", raising=False)

#     with pytest.raises(ValueError):
#         LLM()


# def test_llm_rejects_empty_prompt(monkeypatch):
#     monkeypatch.setenv("OPENAI_API_KEY", "test-key")

#     llm = LLM()

#     with pytest.raises(ValueError):
#         llm.generate("")

from rag.llm import FakeLLM


def test_fake_llm_refund_answer():
    llm = FakeLLM()

    answer = llm.generate(
        """
        Context:
        Customers can request a refund within 30 days.

        Question:
        Can customers request a refund?
        """
    )

    assert "30 days" in answer


def test_fake_llm_unknown_question():
    llm = FakeLLM()

    answer = llm.generate(
        """
        Context:
        The company provides customer support.

        Question:
        What is the employee vacation policy?
        """
    )

    assert answer == "I don't have enough information to answer that."


def test_fake_llm_rejects_empty_prompt():
    llm = FakeLLM()

    try:
        llm.generate("")
        assert False
    except ValueError:
        assert True