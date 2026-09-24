def build_prompt(question: str, context: str) -> str:
    """
    Build a prompt using the user question and retrieved context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not context.strip():
        raise ValueError("Context cannot be empty")

    return f"""
You are a helpful customer support assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I don't have enough information to answer that."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""".strip()