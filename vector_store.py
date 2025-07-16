from typing import List
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Initialize the embedding model
embeddings = OpenAIEmbeddings(chunk_size=50)

# numerical vector representation of a piece of text
# Texts with similar meaning have similar vectors
# Chunk size controls the batch size of documents sent to the embedding API at once
# In this case, up to 50 text chunks are batched together and sent to the OpenAI API at once for embedding


# 2. Build a Chroma vector store from document chunks
def build_vector_store(
        chunks: List[Document],
        collection_name: str = "pmc_002",
        persist_directory: str = "./chroma_store"
) -> Chroma:
    """
    Embeds and stores document chunks in a Chroma vector store
    (Use when first time running pipeline or adding new documents, 
    or changed chunking and embeddings)

    Args:
    chunks: List of chunked Document objects
    collection_name: Name of the Chroma collection (default "pmc_002").
    persist_directory: Directory to persist the collection (default "./chroma_store").

    Returns:
    A Chroma vector store object
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name = collection_name,
        persist_directory=persist_directory
    )
    num_vectors = vectordb._collection.count()
    print(f"[VectorStore] Indexed {num_vectors} vectors into collection '{collection_name}'.")
    return vectordb

# 3. Load an existing Chroma vector store from disk
def load_vector_store(
        collection_name: str = "pmc_002",
        persist_directory: str = "./chroma_store"
) -> Chroma:
    """
    Loads an existing Chroma vector store from disk.
    (Use when needing to lead saved vectors to skip recomputation)

    Args:
    collection_name: Name of the Chroma collection.
    persist_directory: Directory where the collection is stored.

    Returns:
    A chroma vector store object
    """
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    num_vectors = vectordb._collection.count()
    print(f"[VectorStore] Loaded {num_vectors} vectors from '{collection_name}'.")
    return vectordb

# 4. Return a retriever interface from the vector store
def get_retriever(vectordb: Chroma, k: int = 5):
    """
    Converts a Chroma vector store into a retriever for RAG.
    Allows Langchain to use and do semantic similarity search

    Args:
    vectordb: A Chroma vector store object.
    k: Number of relevant documents to retrieve (default 5).
    Will make it to where in gradio UI, I can change it

    Returns:
    A retriever object to use with LangChain chains.
    """
    return vectordb.as_retriever(search_kwargs={"k": k})