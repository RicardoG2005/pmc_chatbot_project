import os
import argparse
from env_setup import configure_env
from data_loader import load_documents
from text_processing import chunk_documents
from vector_store import build_vector_store, load_vector_store
from frontend_grad import launch_gradio_app

def build_pipeline(
    folder_path: str,
    force_rebuild: bool = False,
    limit: int = 500,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    collection_name: str = "pmc_002",
    persist_directory: str = "./chroma_store"
):
    '''
    Args:
    folder_path: Path to folder containing XML (NXML) files
    force_rebuild: If True, rebuild vector store from scratch
    limit: Max number of XML files to load
    chunk_size: Character count per chunk 
    chunk_overlap: Overlap between chunks
    collection_name: Name of chroma collection
    persist_directory: where to save Chroma vector DB 
    '''
    configure_env()

    if force_rebuild or not os.path.exists(persist_directory):
        print("[Main] Building new vector store ...")
        docs = load_documents(folder_path, limit = limit)
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        build_vector_store(chunks, collection_name=collection_name, persist_directory=persist_directory)
    else:
        print("[Main] Vector store already exists. Skipping build.")

def main():
    parser = argparse.ArgumentParser("BioMed RAG Chatbot")
    parser.add_argument("--docs", type=str, default="./nxml_data", help="Folder containing NXML files")
    parser.add_argument("--force", action="store_true", help = "Force rebuild of vector store")
    parser.add_argument("--no-ui", action="store_true", help="Skip launching the Gradio UI")
    args = parser.parse_args()

    build_pipeline(folder_path=args.docs, force_rebuild=args.force)

    if not args.no_ui:
        launch_gradio_app()
    
if __name__ == "__main__":
    main()
