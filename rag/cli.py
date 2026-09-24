from rag.ingest import ingest_documents
from rag.retriever import Retriever
from rag.prompt import build_prompt
from rag.llm import FakeLLM


def main():
    print("Loading documents...")

    store = ingest_documents()

    retriever = Retriever(
        store,
        top_k=2,
    )

    llm = FakeLLM()

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

        # Convert question into an embedding
        from rag.embeddings import EmbeddingModel

        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.embed_text(question)

        # Retrieve relevant documents
        result = retriever.retrieve(query_embedding)

        # Get retrieved documents
        documents = result["documents"][0]

        # Get retrieved metadata
        metadatas = result["metadatas"][0]

        # Build context
        context = "\n\n".join(documents)

        # Build prompt
        prompt = build_prompt(
            question=question,
            context=context,
        )

        # Generate answer
        answer = llm.generate(prompt)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        unique_sources = []

        for metadata in metadatas:
            source = metadata["source"]

            if source not in unique_sources:
                unique_sources.append(source)

        for source in unique_sources:
            print(f"- {source}")

        print()


if __name__ == "__main__":
    main()