from env_setup import configure_env
from vector_store import load_vector_store, get_retriever

# Environment setup (entry point for RAG pipeline modules)

# Ensure necessary environment variables such as OPENAI_API_KEY are available
# This is safe to run multiple times
configure_env()

# Vector store and retriever
default_k: int = 5 # default number of documents to retrieve

# Load the persisted Chroma collection once at import time
_vectordb = load_vector_store()

# Expose a ready-made retriever that other modules can import directly
retriever = get_retriever(_vectordb, k = default_k)

# Conversational Retrieval Chain helpers (kept from original file)

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
from langchain.vectorstores import Chroma

# Prompt template
__system_prompt = """
You are a specialist research assistant working with medical information.
You will recieve a message from a user who might have a question or to 
just learn more about a certain topic. 
You will have a knowledge base which is a list of passages (context) from
biomedical articles. 
Always ground your answer in the retrieved context. If the context lacks
information about the message the user asks, say "No info in context, but here
is what I know..." and then answer from your background knowledge. 
"""

_prompt_with_context = ChatPromptTemplate.from_messages(
    [
        ("system", __system_prompt),
        MessagesPlaceholder(variable_name="context"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

def build_rag_chain(
        vectordb: Chroma | None = None,
        *,
        k: int = default_k,
        memory_k: int = 10,
        return_sources: bool = False,
) -> ConversationalRetrievalChain:
    """
    Return a configured ConversationalRetrievalChain

    If *vectordb* is omitted, the module-level Chroma store loaded at import
    time will be used.
    """

    db = vectordb or _vectordb
    _retriever = get_retriever(db, k = k)

    memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        input_key="question",
        k=memory_k,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=return_sources,
        combine_docs_chain_kwargs={"prompt": _prompt_with_context},
    )
    return chain

def ask(
        chain: ConversationalRetrievalChain,
        question: str,
        chat_history: Optional[list[HumanMessage]] = None,
):
    """
    Invoke *chain* with a given *question* and optional *chat_history*.
    """
    payload = {"question": question, "chat_history": chat_history or []}
    reply = chain.invoke(payload)
    return reply["answer"]

# Update the public symbols that get imported via 'from rag_chain import *'
__all__ = [
    "retriever",
    "default_k",
    "build_rag_chain",
    "ask"
]
