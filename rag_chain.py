from env_setup import configure_env
from vector_store import load_vector_store, get_retriever

# -----------------------------------------------------------------------------
# Environment setup (entry-point for RAG pipeline modules)
# -----------------------------------------------------------------------------

# Ensure necessary environment variables such as OPENAI_API_KEY are available
# This is safe to run multiple times; subsequent calls are no-ops.
configure_env()

# -----------------------------------------------------------------------------
# Vector store & retriever
# -----------------------------------------------------------------------------

default_k: int = 5  # default number of documents to retrieve

# Load the persisted Chroma collection once at import time
_vectordb = load_vector_store()

# Expose a ready-made retriever that other modules can import directly
retriever = get_retriever(_vectordb, k=default_k)

__all__ = ["retriever", "default_k"]