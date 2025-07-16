from typing import Optional

# Support both old and new LangChain import paths
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage

from llm_core import model
from vector_store import get_retriever
from langchain.vectorstores import Chroma

__all__ = ["build_rag_chain", "ask"]

system_prompt = """
You are a specialist research assistant working with medical information.
You will receive a message from a user who might have a question or to
just learn more about a certain topic.
You will have a knowledge base which is a list of passages (context) from
biomedical articles.
Always ground your answer in the retrieved context. If the context lacks
information about the message the user asks, say "No info in context, but here 
is what I know..." and then answer from your background knowledge.
"""

prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="context"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])

def build_rag_chain(
        vectordb:Chroma,
        k: int = 5,
        memory_k: int = 10,
        return_sources: bool = False
) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain around the core LLM.

    Args:
    vectordb: A populated Chroma vector store.
    k: Number of documents to retrieve per query.
    memory_k: How many past turns to keep in memory 
    return_sources: Whether to return source docs with the answer

    Returns:
    Configured ConversationalRetrievalChain instance.
    """
    retriever = get_retriever(vectordb, k=k)

    memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        input_key = "question",
        k = memory_k,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents = return_sources,
        combine_docs_chain_kwargs={"prompt": prompt_with_context},
    )
    return chain

def ask(chain: ConversationalRetrievalChain, question: str, chat_history: Optional[list[HumanMessage]] = None):
    """
    Utility helper to invoke the RAG chain

    Args:
    chain: The ConversationalRetrievalChain build by 'build_rag_chain'.
    question: the user's query
    chat_history: Existing chat history to maintain thread continuity.

    Returns:
    the answer string (and optionally source docs if configured).
    """
    payload = {"question": question, "cha_history": chat_history or []}
    reply = chain.invoke(payload)
    return reply["answer"]
