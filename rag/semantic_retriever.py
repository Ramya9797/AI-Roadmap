
class SemanticRetrieverAdapter:
    def __init__(self, retriever, embedding_model):
        self.retriever = retriever
        self.embedding_model = embedding_model

    def search(
        self,
        query: str,
        top_k: int = 3,
        where=None,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedding_model.embed_documents(
            [query]
        )[0]

        results = self.retriever.retrieve(
            query_embedding,
            where=where,
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0]

        output = []

        for index, (document_id, document, distance) in enumerate(
            zip(ids, documents, distances)
        ):
            score = 1.0 / (1.0 + distance)

            result = {
                "id": document_id,
                "text": document,
                "score": score,
            }

            if index < len(metadatas):
                result["metadata"] = metadatas[index]

            output.append(result)

        return output[:top_k]
