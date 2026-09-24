from langchain_ollama import ChatOllama


def create_llm():
    """
    Create a local LangChain chat model using Ollama.
    """

    return ChatOllama(
        model="llama3.2",
        temperature=0,
    )