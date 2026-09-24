from rag.embeddings import EmbeddingModel
from rag.prompt import build_prompt
from rag.retriever import Retriever


class RAGPipeline:
    """
    Connects query embedding, retrieval, prompt construction,
    and LLM generation.
    """

    def __init__(
        self,
        store,
        llm,
        top_k: int = 3,
    ):
        self.embedding_model = EmbeddingModel()
        self.retriever = Retriever(
            store,
            top_k=top_k,
        )
        self.llm = llm

    def answer(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedding_model.embed_documents(
            [question]
        )[0]

        results = self.retriever.retrieve(
            query_embedding
        )

        documents = results["documents"][0]

        if not documents:
            return "I don't have enough information to answer that."

        context = "\n\n".join(documents)

        prompt = build_prompt(
            question=question,
            context=context,
        )

        return self.llm.generate(prompt)