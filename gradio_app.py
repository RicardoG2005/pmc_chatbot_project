"""Gradio interface for the PMC RAG chatbot.

Run this module to launch a local web UI:
    python gradio_app.py

Make sure you have first created or loaded the Chroma vector store.
You can set the environment variable PMC_XML_DIR to the folder with NXML
files if you need to build the store automatically on first run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gradio as gr
from langchain.schema import HumanMessage

from env_setup import configure_env
from data_loader import load_documents
from text_processing import chunk_documents
from vector_store import load_vector_store, build_vector_store
from rag_chain import build_rag_chain, ask

# ----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# ----------------------------------------------------------------------------

COLLECTION_NAME = "pmc_002"
PERSIST_DIRECTORY = "./chroma_store"

# Folder containing raw NXML files.  Used only if we need to build a store.
PMC_XML_DIR = os.getenv("PMC_XML_DIR")  # e.g. "/data/pmc/PMC002xxxxxx"

# Retrieval parameters
K_RETRIEVE = int(os.getenv("PMC_K", 5))  # default top-k
MEMORY_K = int(os.getenv("PMC_MEMORY_K", 10))
RETURN_SOURCES = bool(int(os.getenv("PMC_RETURN_SOURCES", 0)))

# ----------------------------------------------------------------------------
# Prepare vector store & chain -------------------------------------------------
# ----------------------------------------------------------------------------

configure_env()

try:
    vectordb = load_vector_store(collection_name=COLLECTION_NAME,
                                 persist_directory=PERSIST_DIRECTORY)
except Exception as err:
    print(f"[Warning] Could not load existing vector store: {err}")
    if PMC_XML_DIR:
        print("[Info] Building a new vector store from raw documents …")
        docs = load_documents(PMC_XML_DIR, limit=500)
        chunks = chunk_documents(docs)
        vectordb = build_vector_store(chunks,
                                      collection_name=COLLECTION_NAME,
                                      persist_directory=PERSIST_DIRECTORY)
    else:
        raise RuntimeError("Vector store not found and PMC_XML_DIR not set.") from err

rag_chain = build_rag_chain(vectordb,
                            k=K_RETRIEVE,
                            memory_k=MEMORY_K,
                            return_sources=RETURN_SOURCES)

# ----------------------------------------------------------------------------
# Gradio Interface -------------------------------------------------------------
# ----------------------------------------------------------------------------

def respond(message: str, history: list[list[str]]):
    """Gradio chat callback.

    Parameters
    ----------
    message : str
        The user's latest message.
    history : list[list[str]]
        Prior turns as [[user, bot], …]. The chain keeps its own memory, so
        we don't have to convert history into LangChain format; we simply
        ask the chain for a reply and let its memory track context.
    """
    answer = ask(rag_chain, message)
    return answer


def launch_ui(server_name: Optional[str] = None, server_port: Optional[int] = None):
    """Launches the Gradio web application."""
    demo = gr.ChatInterface(
        fn=respond,
        title="PMC Biomedical Chatbot (RAG)",
        description=(
            "Ask biomedical research questions. The assistant retrieves relevant "
            "passages from the PubMed Central corpus and grounds its answers in "
            "those sources."
        ),
        theme="soft",
    )
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_ui()