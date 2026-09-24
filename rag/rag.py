# from rag.embeddings import EmbeddingModel
# from rag.prompt import build_prompt
# from rag.retriever import Retriever


# class RAGPipeline:
#     """
#     Connects query embedding, retrieval, prompt construction,
#     and LLM generation.
#     """

#     def __init__(
#         self,
#         store,
#         llm,
#         top_k: int = 3,
#     ):
#         self.embedding_model = EmbeddingModel()
#         self.retriever = Retriever(
#             store,
#             top_k=top_k,
#         )
#         self.llm = llm

#     def answer(self, question: str, where=None) -> str:
#         """
#         Return only the generated answer.
#         """
#         result = self.answer_with_citations(
#             question,
#             where=where,
#         )
#         return result["answer"]

#     def answer_with_citations(self, question: str, where=None) -> dict:
#         """
#         Generate an answer together with citations from
#         the retrieved document chunks.
#         """

#         if not question.strip():
#             raise ValueError("Question cannot be empty")

#         query_embedding = self.embedding_model.embed_documents(
#             [question]
#         )[0]

#         if where is None:
#             results = self.retriever.retrieve(query_embedding)
#         else:
#             results = self.retriever.retrieve(
#                 query_embedding,
#                 where=where,
#         )


#         documents = results["documents"][0]
#         metadatas = results["metadatas"][0]

#         if not documents:
#             return {
#                 "answer": "I don't have enough information to answer that.",
#                 "citations": [],
#             }

#         context = "\n\n".join(documents)

#         prompt = build_prompt(
#             question=question,
#             context=context,
#         )

#         answer = self.llm.generate(prompt)

#         citations = []

#         for document, metadata in zip(
#             documents,
#             metadatas,
#         ):
#             citations.append(
#                 {
#                     "source": metadata["source"],
#                     "chunk_id": metadata["chunk_id"],
#                     "text": document,
#                 }
#             )

#         return {
#             "answer": answer,
#             "citations": citations,
#         }


from rag.embeddings import EmbeddingModel
from rag.prompt import build_prompt
from rag.retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        store,
        llm,
        top_k: int = 3,
        query_rewriter=None,
    ):
        self.embedding_model = EmbeddingModel()
        self.retriever = Retriever(store, top_k=top_k)
        self.llm = llm
        self.query_rewriter = query_rewriter

    def answer(self, question: str, where=None) -> str:
        result = self.answer_with_citations(
            question,
            where=where,
        )
        return result["answer"]

    def answer_with_citations(self, question: str, where=None) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        normalized_question = question.strip()

        query_rewriter = getattr(
            self,
            "query_rewriter",
            None,
        )

        if query_rewriter is not None:
            search_query = query_rewriter.rewrite(
                normalized_question
            )
        else:
            search_query = normalized_question

        query_embedding = self.embedding_model.embed_documents(
            [search_query]
        )[0]

        if where is None:
            results = self.retriever.retrieve(query_embedding)
        else:
            results = self.retriever.retrieve(
                query_embedding,
                where=where,
            )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            return {
                "answer": "I don't have enough information to answer that.",
                "citations": [],
            }

        context = "\n\n".join(documents)

        prompt = build_prompt(
            question=question,
            context=context,
        )

        answer = self.llm.generate(prompt)

        citations = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):
            citations.append({
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "text": document,
            })

        return {
            "answer": answer,
            "citations": citations,
        }
