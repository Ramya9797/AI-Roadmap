from typing import Dict, List

import chromadb


DEFAULT_COLLECTION_NAME = "rag_documents"


class ChromaVectorStore:
    """
    Simple wrapper around ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> None:
        """
        Store chunks, embeddings and metadata in Chroma.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "document_type": chunk["document_type"],
                "document_name": chunk["document_name"],
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        """
        Return number of stored chunks.
        """

        return self.collection.count()

    def get_all(self) -> Dict:
        """
        Return all stored records.
        """

        return self.collection.get()


    def search(
    self,
    query_embedding: List[float],
    top_k: int = 3,
    ) -> Dict:
        """
        Search Chroma for the most similar chunks.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                self.collection.count(),
            ),
        )

        return results