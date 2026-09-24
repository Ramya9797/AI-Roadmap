from rag.ingest import ingest_documents
from rag.rag import RAGPipeline
from rag.llm import FakeLLM


def main():
    print("Building RAG index...")

    store = ingest_documents()

    pipeline = RAGPipeline(
        store=store,
        llm=FakeLLM(),
        top_k=2,
    )

    test_cases = [
        {
            "question": "Can customers request a refund?",
            "expected": "30 days",
        },
        {
            "question": "How should order status be retrieved?",
            "expected": "order tracking",
        },
        {
            "question": "What should support teams do for repeated unresolved issues?",
            "expected": "escalate",
        },
        {
            "question": "What generates the customer churn probability?",
            "expected": "churn prediction",
        },
    ]

    passed = 0

    print("\nStarting RAG answer evaluation...\n")

    for test_case in test_cases:
        question = test_case["question"]
        expected = test_case["expected"]

        answer = pipeline.answer(question)

        success = expected.lower() in answer.lower()

        if success:
            passed += 1

        print("=" * 60)
        print(f"Question: {question}")
        print(f"Expected keyword: {expected}")
        print(f"Answer: {answer}")
        print(f"Result: {'PASS' if success else 'FAIL'}")

    total = len(test_cases)

    print("\n" + "=" * 60)
    print("RAG ANSWER EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {(passed / total) * 100:.1f}%")


if __name__ == "__main__":
    main()