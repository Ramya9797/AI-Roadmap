from typing import Dict, List

from rag.vector_store import ChromaVectorStore


class Retriever:
    """
    Retrieves relevant document chunks from Chroma.
    """

    def __init__(
        self,
        store: ChromaVectorStore,
        top_k: int = 3,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query_embedding: List[float],
    ) -> Dict:
        """
        Retrieve the most relevant chunks.
        """

        return self.store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )