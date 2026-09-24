from llm.client import create_llm
from llm.prompts import create_qa_prompt


def create_qa_chain():
    """
    Create a LangChain pipeline:
    
    PromptTemplate → LLM
    """

    prompt = create_qa_prompt()
    llm = create_llm()

    chain = prompt | llm

    return chain