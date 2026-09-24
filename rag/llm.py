# import pytest

# from rag.llm import FakeLLM


# def test_fake_llm_returns_answer():
#     llm = FakeLLM()

#     answer = llm.generate(
#         "What is the refund period?"
#     )

#     assert isinstance(answer, str)
#     assert answer != ""


# def test_fake_llm_returns_expected_mock_answer():
#     llm = FakeLLM()

#     answer = llm.generate(
#         "What is the refund period?"
#     )

#     assert answer == (
#         "This is a mock answer generated from the provided context."
#     )


# def test_fake_llm_rejects_empty_prompt():
#     llm = FakeLLM()

#     with pytest.raises(ValueError):
#         llm.generate("")import pytest

# from rag.llm import FakeLLM


# def test_fake_llm_returns_answer():
#     llm = FakeLLM()

#     answer = llm.generate(
#         "What is the refund period?"
#     )

#     assert isinstance(answer, str)
#     assert answer != ""


# def test_fake_llm_returns_expected_mock_answer():
#     llm = FakeLLM()

#     answer = llm.generate(
#         "What is the refund period?"
#     )

#     assert answer == (
#         "This is a mock answer generated from the provided context."
#     )


# def test_fake_llm_rejects_empty_prompt():
#     llm = FakeLLM()

#     with pytest.raises(ValueError):
#         llm.generate("")

class FakeLLM:
    """
    Mock LLM used for testing the RAG pipeline
    without requiring an API key.
    """

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt_lower = prompt.lower()

        if "refund" in prompt_lower:
            return "Customers can request a refund within 30 days."

        if "order status" in prompt_lower:
            return "Order status can be retrieved using the order tracking system."

        if "churn probability" in prompt_lower:
            return "Customer churn probability is generated using the churn prediction model."

        if "repeated unresolved issues" in prompt_lower:
            return "Support teams should escalate repeated unresolved issues for further investigation."

        return "I don't have enough information to answer that."