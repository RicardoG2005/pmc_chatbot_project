"""Main entry-point for the PMC Chatbot project.

Usage:
    python main.py index   # parse NXML files, chunk, build vector DB
    python main.py gradio  # launch the Gradio web chat interface
    python main.py cli     # simple terminal chat (optional)

Before running any command make sure you have:
    1. A `.env` file with OPENAI_API_KEY=<your key>
    2. Downloaded a set of PMC NXML files in a folder (default ./pmc_xml)

"""
from __future__ import annotations

import argparse
import sys

from env_setup import configure_env
from data_loader import load_documents
from text_processing import chunk_documents
from vector_store import build_vector_store, load_vector_store
from rag_chain import retriever, default_k  # ensures vector store & model are ready

# Optional: CLI chat support
from llm_core import chat

# -----------------------------------------------------------------------------
# Constants – adjust paths/collection names here if needed
# -----------------------------------------------------------------------------

PMC_XML_FOLDER = "./pmc_xml"          # folder containing *.nxml or *.xml files
CHUNK_COLLECTION = "pmc_002"         # Chroma collection name
CHROMA_DIR = "./chroma_store"         # Disk location for Chroma
MAX_FILES = 500                        # limit when building the store

# -----------------------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------------------

def run_index(limit: int = MAX_FILES):
    """Parse XML files, chunk text and build / overwrite the Chroma vector store."""
    docs = load_documents(PMC_XML_FOLDER, limit=limit)
    chunks = chunk_documents(docs)
    build_vector_store(chunks, collection_name=CHUNK_COLLECTION, persist_directory=CHROMA_DIR)


def run_gradio():
    """Launch the Gradio web UI defined in gradio_app.py."""
    import gradio_app  # local import so that heavy deps are only loaded if needed
    gradio_app.main()


def run_cli():
    """Very small interactive loop in the terminal (no UI)."""
    print("\nEnter 'exit' to quit. Typing begins…\n")
    thread_id = "cli"
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        answer = chat(q, thread_id=thread_id)
        print(f"Assistant: {answer}\n")

# -----------------------------------------------------------------------------
# Argument parsing & dispatch
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PMC Chatbot pipeline & interface")
    sub = p.add_subparsers(dest="command", required=True)

    # index sub-command
    p_index = sub.add_parser("index", help="Load XML files, create / update the vector store")
    p_index.add_argument("--limit", type=int, default=MAX_FILES, help="Maximum number of XML files to process")

    # gradio sub-command
    sub.add_parser("gradio", help="Launch Gradio chat UI")

    # cli sub-command
    sub.add_parser("cli", help="Run simple terminal chat loop")

    return p

# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    # Ensure environment variables are available early
    configure_env()

    args = build_arg_parser().parse_args(argv)

    if args.command == "index":
        run_index(limit=args.limit)
    elif args.command == "gradio":
        run_gradio()
    elif args.command == "cli":
        run_cli()
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()