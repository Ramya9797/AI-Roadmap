# from rag.ingest import ingest_documents
# from rag.retriever import Retriever
# from rag.embeddings import EmbeddingModel


# def main():
#     print("Building RAG index...")

#     store = ingest_documents()

#     retriever = Retriever(
#         store,
#         top_k=2,
#     )

#     embedding_model = EmbeddingModel()

#     questions = [
#         "Can customers request a refund?",
#         "How should order status be retrieved?",
#         "What should support teams do for repeated unresolved issues?",
#         "What generates the customer churn probability?",
#     ]

#     for question in questions:
#         print("\n" + "=" * 60)
#         print(f"Question: {question}")

#         query_embedding = embedding_model.embed_text(question)

#         result = retriever.retrieve(query_embedding)

#         print("\nRetrieved documents:")

#         for i, document in enumerate(result["documents"][0], start=1):
#             print(f"\n{i}. {document}")

#         print("\nSources:")

#         for metadata in result["metadatas"][0]:
#             print(metadata["source"])


# if __name__ == "__main__":
#     main()

from rag.ingest import ingest_documents
from rag.retriever import Retriever
from rag.embeddings import EmbeddingModel


def main():
    print("Building RAG index...")

    store = ingest_documents()

    retriever = Retriever(
        store,
        top_k=2,
    )

    embedding_model = EmbeddingModel()

    test_cases = [
        {
            "question": "Can customers request a refund?",
            "expected_source": "refund_policy.md",
        },
        {
            "question": "How should order status be retrieved?",
            "expected_source": "order_policy.md",
        },
        {
            "question": "What should support teams do for repeated unresolved issues?",
            "expected_source": "customer_support.md",
        },
        {
            "question": "What generates the customer churn probability?",
            "expected_source": "churn_strategy.md",
        },
    ]

    passed = 0

    print("\nStarting retrieval evaluation...\n")

    for test_case in test_cases:
        question = test_case["question"]
        expected_source = test_case["expected_source"]

        query_embedding = embedding_model.embed_text(question)

        result = retriever.retrieve(query_embedding)

        sources = [
            metadata["source"]
            for metadata in result["metadatas"][0]
        ]

        success = expected_source in sources

        if success:
            passed += 1

        print("=" * 60)
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {sources}")
        print(f"Result: {'PASS' if success else 'FAIL'}")

    total = len(test_cases)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {(passed / total) * 100:.1f}%")


if __name__ == "__main__":
    main()