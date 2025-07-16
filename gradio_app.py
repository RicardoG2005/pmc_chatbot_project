import os
import uuid
from typing import List, Tuple

import gradio as gr
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from env_setup import configure_env
from llm_core import model, system_text  # reuse the same model & prompt basis
from vector_store import load_vector_store, get_retriever

# -----------------------------------------------------------------------------
# Environment & global resources
# -----------------------------------------------------------------------------

# Ensure .env is loaded and OPENAI_API_KEY is set
configure_env()

# Load the persisted Chroma collection once at start-up.
# You may need to run the indexing pipeline first (see data_loader / text_processing).
vectordb = load_vector_store()

# Default number of documents to retrieve. Can be changed from the UI.
default_k = 5
retriever = get_retriever(vectordb, k=default_k)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _build_prompt(question: str, context_chunks: List[str], chat_history: List[Tuple[str, str]]):
    """Convert query, context & history to a list of LangChain messages."""
    messages: List[SystemMessage | HumanMessage | AIMessage] = []

    # 1. System message – the same text as llm_core uses
    messages.append(SystemMessage(system_text.strip()))

    # 2. Add retrieved context as separate human message so that the model can ground its answer
    if context_chunks:
        joined_context = "\n\n".join(context_chunks)
        context_msg = f"Here are relevant context passages:\n\n{joined_context}"
        messages.append(HumanMessage(content=context_msg))

    # 3. Add previous turns (alternating user/assistant) so the model keeps history
    for user_turn, assistant_turn in chat_history:
        messages.append(HumanMessage(content=user_turn))
        messages.append(AIMessage(content=assistant_turn))

    # 4. Finally, the new user question
    messages.append(HumanMessage(content=question))
    return messages


def _answer(question: str, history: List[Tuple[str, str]], k: int):
    """Retrieve context, call the model and return the answer string."""
    docs = retriever.invoke(question, k=k) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(question)
    context_chunks = [d.page_content for d in docs]

    messages = _build_prompt(question, context_chunks, history)
    response: AIMessage = model.invoke(messages)
    return response.content, context_chunks


def user_asks(message: str, history: List[Tuple[str, str]], k_slider: int):
    """Gradio event handler for a single user query."""
    response, _ = _answer(message, history, k_slider)
    history.append((message, response))
    return "", history  # clear textbox, update chatbot


def regenerate(last_user_message: str, history: List[Tuple[str, str]], k_slider: int):
    """Regenerate assistant reply for the last user message."""
    # Pop the previous assistant response
    if history:
        last_user, _ = history.pop()  # remove last pair
        # Ensure we use the provided last_user_message if available
        question = last_user_message or last_user
    else:
        question = last_user_message

    answer, _ = _answer(question, history, k_slider)
    history.append((question, answer))
    return history

# -----------------------------------------------------------------------------
# Build Gradio Interface
# -----------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="BioMed RAG Chat") as demo:
        gr.Markdown("""# 🩺 BioMed RAG Chat\nAsk biomedical questions and get answers grounded in primary literature.""")

        chatbot = gr.Chatbot(height=500, label="Chat")
        user_input = gr.Textbox(placeholder="Ask a question…", label="Your question")
        k_slider = gr.Slider(minimum=1, maximum=10, value=default_k, step=1, label="Documents to retrieve (k)")

        send_btn = gr.Button("Send")
        regen_btn = gr.Button("Regenerate")

        send_btn.click(user_asks, inputs=[user_input, chatbot, k_slider], outputs=[user_input, chatbot])
        regen_btn.click(regenerate, inputs=[user_input, chatbot, k_slider], outputs=[chatbot])

        gr.Markdown("""---\n**Note:** This application retrieves passages from a vector store created from PubMed Central articles (sample set).""")

    return demo


def main():
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()