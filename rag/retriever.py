from datetime import datetime
from typing import Dict, List

from rag.recency import calculate_recency_score
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
        query_embedding,
        where=None,
        use_recency=False,
    ):
        result = self.store.search(
            query_embedding,
            top_k=self.top_k,
            where=where,
        )

        if not use_recency:
            return result

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        now = datetime.now()

        ranked_results = []

        for document, metadata in zip(documents, metadatas):
            created_at = metadata.get("created_at")

            if created_at:
                created_at = datetime.fromisoformat(created_at)
                recency_score = calculate_recency_score(
                    created_at=created_at,
                    now=now,
                )
            else:
                recency_score = 0.0

            ranked_results.append(
                (
                    recency_score,
                    document,
                    metadata,
                )
            )

        ranked_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return {
            "documents": [[
                item[1]
                for item in ranked_results
            ]],
            "metadatas": [[
                item[2]
                for item in ranked_results
            ]],
        }