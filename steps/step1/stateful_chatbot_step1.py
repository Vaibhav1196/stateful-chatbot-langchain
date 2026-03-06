"""Step 1: Basic stateful chatbot using per-session in-memory history."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env).")

model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# session_id -> chat history (hash map based lookup)
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return an existing session history or create a new one."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chat = RunnableWithMessageHistory(model, get_session_history)


def chat_once(session_id: str, text: str) -> str:
    """Send one user message and return the assistant response."""
    config = {"configurable": {"session_id": session_id}}
    response = chat.invoke([HumanMessage(content=text)], config=config)
    return response.content


if __name__ == "__main__":
    session_id = input("Session ID (default: chat1): ").strip() or "chat1"
    print(f"\nUsing session: {session_id}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("Bot:", chat_once(session_id, user_input), "\n")
