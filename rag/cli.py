from rag.ingest import ingest_documents
from rag.rag import RAGPipeline
from rag.llm import FakeLLM


def format_citations(citations):
    """
    Format citation information for display.
    """

    if not citations:
        return "No sources available."

    lines = []

    for index, citation in enumerate(citations, start=1):
        source = citation["source"]
        chunk_id = citation["chunk_id"]
        text = citation["text"]

        lines.append(f"[{index}] {source}")
        lines.append(f"Chunk: {chunk_id}")
        lines.append(f'"{text}"')
        lines.append("")

    return "\n".join(lines).strip()


def main():
    print("Loading documents...")

    store = ingest_documents()

    llm = FakeLLM()

    pipeline = RAGPipeline(
        store,
        llm,
        top_k=2,
    )

    print("RAG system ready.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        result = pipeline.answer_with_citations(
            question
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        print(format_citations(result["citations"]))

        print()


if __name__ == "__main__":
    main()