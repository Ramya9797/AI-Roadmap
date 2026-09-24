from llm.prompts import create_qa_prompt


def test_qa_prompt_creation():
    prompt = create_qa_prompt()

    messages = prompt.invoke(
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "Can customers request a refund?",
        }
    )

    assert len(messages.messages) == 2


def test_qa_prompt_contains_context():
    prompt = create_qa_prompt()

    messages = prompt.invoke(
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "Can customers request a refund?",
        }
    )

    formatted_prompt = str(messages)

    assert "Customers can request a refund within 30 days." in formatted_prompt
    assert "Can customers request a refund?" in formatted_prompt


def test_qa_prompt_contains_system_instruction():
    prompt = create_qa_prompt()

    messages = prompt.invoke(
        {
            "context": "Test context",
            "question": "Test question",
        }
    )

    formatted_prompt = str(messages)

    assert "helpful AI assistant" in formatted_prompt