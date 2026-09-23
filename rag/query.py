from rag.embeddings import EmbeddingModel
from rag.ingest import ingest_documents
from rag.retriever import Retriever


def search_documents(
    question: str,
    top_k: int = 3,
):
    """
    Search the vector database using a natural-language question.
    """

    print("Loading embedding model...")

    model = EmbeddingModel()

    print("Ingesting documents...")

    store = ingest_documents()

    retriever = Retriever(
        store,
        top_k=top_k,
    )

    print("Creating query embedding...")

    query_embedding = model.embed_documents(
        [question]
    )[0]

    print("Searching Chroma...")

    results = retriever.retrieve(
        query_embedding
    )

    return results


if __name__ == "__main__":

    question = "How can we reduce customer churn?"

    results = search_documents(
        question,
        top_k=3,
    )

    print()
    print("QUESTION:")
    print(question)

    print()
    print("RETRIEVED DOCUMENTS:")

    for document in results["documents"][0]:
        print("--------------------------------")
        print(document)