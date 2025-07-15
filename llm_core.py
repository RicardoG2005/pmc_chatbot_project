import os
from env_setup import configure_env, set_env_var

# Ensure .env is loaded and keys are set
configure_env()

import getpass
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# 1. Initialize the LLM
def _init_model():
    # if OPENAI_API_KEY still missing, prompt for it
    set_env_var("OPENAI_API_KEY", "Enter your OpenAI API key: ")
    return init_chat_model("gpt-4o-mini", model_provider="openai")

model = _init_model()

# 2. Trimmer to fit into token budget
trimmer = trim_messages(
    max_tokens = 200,
    strategy = "last",
    token_counter = model,
    include_system = True,
    allow_partial = False,
    start_on = "human",
)

# 3. Prompt template for chat
system_text = """
You are a specialist research assistant working with medical information.
You will receive a message from a user who might have a question or to
just learn more about a certain topic.
You will have a knowledge base which is a list of passages (context) from
biomedical articles.
Always ground your answer in the retrieved context. If the context lacks
information about the message the user asks, say "No info in context, but here 
is what I know..." and then answer from your background knowledge.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    MessagesPlaceholder(variable_name="context"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("user", "{question}"), # Why did we change to context
])

# 4. Build LangGraph-based chat workflow
def build_chat_app() -> StateGraph:
    workflow = StateGraph(state_schema = MessagesState)

    def call_model(state: MessagesState) -> dict:
        # trim history
        trimmed = trimmer.invoke(state["messages"])
        # format prompt
        messages = prompt.format_prompt(
            messages = trimmed
        ).to_messages()
        # invoke LLM
        response = model.invoke(messages)
        return {"messages": response}
    
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    return workflow.compile(checkpointer=MemorySaver())

app = build_chat_app()

# 5. Convenience functions
def chat(query: str, thread_id: str = "default") -> str:
    """
    Send a single query; returns the assistant's full reply
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(query)]}
    output = app.invoke(state, config)
    return output["messages"][-1].content

def stream_chat(query: str, thread_id: str = "default"):
    """
    Stream the assistant's reply token-by-token as a generator.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(query)]}
    for chunk, _ in app.stream(state, config, stream_mode = "messages"):
        if isinstance(chunk, AIMessage):
            yield chunk.content