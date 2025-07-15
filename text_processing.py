from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(
        docs: List[Document],
        chunk_size: int = 4000,
        chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter

    Args:
    docs: List of LangChain Document objects.
    chunk_size: Target number of characters per chunk (default 4000).
    chunk_overlap: number of characters to overlap between chunks (default 200).

    Returns:
    List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"[TextProcessing] Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

# Now we have a list of smaller Document chunks 
# which are ready to be embedded with OpenAI embeddings and
# be stored in a vector database