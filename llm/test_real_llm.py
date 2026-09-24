from llm.client import create_llm
from llm.prompts import create_qa_prompt


def main():
    llm = create_llm()
    prompt = create_qa_prompt()

    messages = prompt.invoke(
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "Can customers request a refund?",
        }
    )

    response = llm.invoke(messages)

    print("\nLLM RESPONSE:")
    print(response.content)


if __name__ == "__main__":
    main()