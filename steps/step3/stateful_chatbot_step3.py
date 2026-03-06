"""Step 3: Add LRU session eviction with thread-safe store abstraction."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict, deque
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
    """Global memory limits."""

    max_turns: int = 8
    enable_token_trimming: bool = True
    max_tokens: int = 1200
    keep_system: bool = True
    max_sessions_in_memory: int = 200


MEMORY = MemoryConfig()


class SessionHistoryStore:
    """Abstract interface for session history stores."""

    def get(self, session_id: str) -> Deque[BaseMessage]:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryLRUHistoryStore(SessionHistoryStore):
    """Thread-safe LRU cache where each value is a bounded message deque."""

    def __init__(self, max_sessions: int, per_session_maxlen: int):
        self.max_sessions = max_sessions
        self.per_session_maxlen = per_session_maxlen
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, Deque[BaseMessage]]" = OrderedDict()

    def get(self, session_id: str) -> Deque[BaseMessage]:
        with self._lock:
            if session_id in self._lru:
                self._lru.move_to_end(session_id, last=True)
                return self._lru[session_id]

            history = deque(maxlen=self.per_session_maxlen)
            self._lru[session_id] = history
            self._lru.move_to_end(session_id, last=True)

            while len(self._lru) > self.max_sessions:
                self._lru.popitem(last=False)

            return history

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._lru.pop(session_id, None)

    def debug_sessions(self) -> list[str]:
        with self._lock:
            return list(self._lru.keys())


store = InMemoryLRUHistoryStore(
    max_sessions=MEMORY.max_sessions_in_memory,
    per_session_maxlen=2 * MEMORY.max_turns,
)


def apply_token_trimming(messages: list[BaseMessage]) -> list[BaseMessage]:
    if not MEMORY.enable_token_trimming:
        return messages
    return trim_messages(
        messages,
        max_tokens=MEMORY.max_tokens,
        strategy="last",
        token_counter=model,
        include_system=MEMORY.keep_system,
    )


class DequeMessageHistory:
    """LangChain history adapter backed by deque."""

    def __init__(self, dq: Deque[BaseMessage]):
        self._dq = dq

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._dq)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self._dq.append(message)


def get_session_history(session_id: str) -> DequeMessageHistory:
    return DequeMessageHistory(store.get(session_id))


chat = RunnableWithMessageHistory(model, get_session_history)


def chat_once(session_id: str, text: str) -> str:
    history = store.get(session_id)

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
    print(f"Per-session window: last {MEMORY.max_turns} turns")
    print(f"Session LRU cap: {MEMORY.max_sessions_in_memory} sessions")
    print(
        f"Token trimming: {'ON' if MEMORY.enable_token_trimming else 'OFF'} "
        f"(max_tokens={MEMORY.max_tokens})"
    )
    print("Commands: /sessions, /delete <session_id>, exit\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        if user_text == "/sessions":
            print("Active sessions in LRU order (oldest -> newest):")
            print(store.debug_sessions(), "\n")
            continue

        if user_text.startswith("/delete"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 2:
                store.delete(parts[1].strip())
                print("Deleted.\n")
            else:
                print("Usage: /delete <session_id>\n")
            continue

        print("Bot:", chat_once(session_id, user_text), "\n")
