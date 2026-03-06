"""Step 2: Bounded memory with deque windowing plus optional token trimming."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env).")

model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)


@dataclass(frozen=True)
class MemoryConfig:
    """Memory policy for one session."""

    max_turns: int = 8
    enable_token_trimming: bool = True
    max_tokens: int = 1200
    keep_system: bool = True


MEMORY = MemoryConfig(max_turns=2, enable_token_trimming=True, max_tokens=1200)

# session_id -> bounded deque of messages
store: dict[str, Deque[BaseMessage]] = {}


def get_or_create_history(session_id: str) -> Deque[BaseMessage]:
    """Create bounded history for a session if needed."""
    if session_id not in store:
        store[session_id] = deque(maxlen=2 * MEMORY.max_turns)
    return store[session_id]


def apply_token_trimming(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Trim old messages until the configured token budget is met."""
    if not MEMORY.enable_token_trimming:
        return messages

    # `token_counter=model` uses model token counting; for Groq this path can
    # require `transformers`, which is included in project dependencies.
    return trim_messages(
        messages,
        max_tokens=MEMORY.max_tokens,
        strategy="last",
        token_counter=model,
        include_system=MEMORY.keep_system,
    )


class DequeMessageHistory:
    """Adapter for RunnableWithMessageHistory using a deque backend."""

    def __init__(self, dq: Deque[BaseMessage]):
        self._dq = dq

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._dq)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self._dq.append(message)


def get_session_history(session_id: str) -> DequeMessageHistory:
    return DequeMessageHistory(get_or_create_history(session_id))


chat = RunnableWithMessageHistory(model, get_session_history)


def chat_once(session_id: str, text: str) -> str:
    """Run one turn with bounded memory controls before and after response."""
    history = get_or_create_history(session_id)

    pre_trimmed = apply_token_trimming(list(history))
    history.clear()
    history.extend(pre_trimmed)

    config = {"configurable": {"session_id": session_id}}
    response = chat.invoke([HumanMessage(content=text)], config=config)

    post_trimmed = apply_token_trimming(list(history))
    history.clear()
    history.extend(post_trimmed)

    return response.content


if __name__ == "__main__":
    session_id = input("Session ID (default: chat1): ").strip() or "chat1"
    print(f"\nUsing session: {session_id}")
    print(f"Memory window: last {MEMORY.max_turns} turns")
    print(
        f"Token trimming: {'enabled' if MEMORY.enable_token_trimming else 'disabled'} "
        f"(max_tokens={MEMORY.max_tokens})"
    )
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        print("Bot:", chat_once(session_id, user_text), "\n")
