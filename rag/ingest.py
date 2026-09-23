from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.metadata import add_metadata
from rag.embeddings import EmbeddingModel
from rag.vector_store import ChromaVectorStore


def ingest_documents():
    """
    Complete document ingestion pipeline.

    Flow:

    Documents
        ↓
    Chunks
        ↓
    Metadata
        ↓
    Embeddings
        ↓
    Chroma
    """

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")
    print()

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks")
    print()

    print("Adding metadata...")

    chunks = add_metadata(chunks)

    print("Metadata added")
    print()

    print("Creating embeddings...")

    model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.embed_documents(texts)

    print(f"Created {len(embeddings)} embeddings")
    print()

    print("Storing in Chroma...")

    store = ChromaVectorStore()

    store.add_chunks(
        chunks,
        embeddings,
    )

    print(f"Stored {store.count()} vectors")
    print()

    print("INGESTION COMPLETE")

    return store


if __name__ == "__main__":
    ingest_documents()