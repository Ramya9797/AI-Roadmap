from langchain_core.prompts import ChatPromptTemplate


def create_qa_prompt():
    """
    Create a reusable LangChain prompt for question answering.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. "
                "Answer questions using only the provided context. "
                "If the answer is not available in the context, "
                "say that you do not have enough information."
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question:\n{question}"
            ),
        ]
    )