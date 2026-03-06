"""Step 4: Prompt discipline with two-layer memory (recent + summary)."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env).")

model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for recent window, summary, and LRU session bounds."""

    max_turns: int = 8
    enable_token_trimming: bool = True
    max_tokens_recent: int = 1200
    summary_max_chars: int = 1200
    summarize_every_n_turns: int = 6
    max_sessions_in_memory: int = 200


MEMORY = MemoryConfig()

SYSTEM_PROMPT = """You are a helpful assistant.
Follow these rules:
- Be accurate and concise.
- If uncertain, say so and ask a focused question.
- Prefer stable user facts/preferences for memory.
- Never retain secrets (keys/passwords) or highly sensitive data.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", "Conversation summary so far (may be incomplete):\n{summary}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | model


@dataclass
class SessionState:
    """Per-session memory state."""

    summary: str
    recent: Deque[BaseMessage]
    turn_count: int = 0


class InMemoryLRUSessionStateStore:
    """Thread-safe LRU store of session states."""

    def __init__(self, max_sessions: int, recent_maxlen: int):
        self.max_sessions = max_sessions
        self.recent_maxlen = recent_maxlen
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, SessionState]" = OrderedDict()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id in self._lru:
                self._lru.move_to_end(session_id, last=True)
                return self._lru[session_id]

            state = SessionState(summary="", recent=deque(maxlen=self.recent_maxlen), turn_count=0)
            self._lru[session_id] = state
            self._lru.move_to_end(session_id, last=True)

            while len(self._lru) > self.max_sessions:
                self._lru.popitem(last=False)

            return state

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._lru.pop(session_id, None)

    def debug_sessions(self) -> list[str]:
        with self._lock:
            return list(self._lru.keys())


store = InMemoryLRUSessionStateStore(
    max_sessions=MEMORY.max_sessions_in_memory,
    recent_maxlen=2 * MEMORY.max_turns,
)


def trim_recent(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Token-bound trim for recent transcript memory."""
    if not MEMORY.enable_token_trimming:
        return messages
    return trim_messages(
        messages,
        max_tokens=MEMORY.max_tokens_recent,
        strategy="last",
        token_counter=model,
        include_system=True,
    )


SUMMARY_SYSTEM = """You summarize conversations into durable memory.

Rules:
- Keep stable facts, preferences, and long-term goals.
- Drop temporary chatter and one-off details.
- Never include secrets or highly sensitive information.
- Keep output short and actionable.
"""


def update_summary(old_summary: str, recent_messages: list[BaseMessage]) -> str:
    """Update long-term summary using previous summary + recent transcript."""
    summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM),
            (
                "human",
                "OLD SUMMARY:\n{old}\n\nNEW MESSAGES:\n{new}\n\nWrite an UPDATED SUMMARY:",
            ),
        ]
    )

    formatted_lines: list[str] = []
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            role = "USER"
        elif isinstance(message, AIMessage):
            role = "ASSISTANT"
        else:
            role = "OTHER"
        formatted_lines.append(f"{role}: {message.content}")

    new_messages_block = "\n".join(formatted_lines)
    summary_response = (summarizer_prompt | model).invoke({"old": old_summary, "new": new_messages_block})

    summary = summary_response.content.strip()
    if len(summary) > MEMORY.summary_max_chars:
        summary = summary[: MEMORY.summary_max_chars].rstrip() + "..."
    return summary


class RecentHistoryAdapter:
    """History adapter for RunnableWithMessageHistory."""

    def __init__(self, state: SessionState):
        self.state = state

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self.state.recent)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self.state.recent.append(message)


def get_session_history(session_id: str) -> RecentHistoryAdapter:
    return RecentHistoryAdapter(store.get(session_id))


chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat_once(session_id: str, text: str) -> str:
    """Single turn with recent-window trim plus periodic summary refresh."""
    state = store.get(session_id)

    pre_trimmed = trim_recent(list(state.recent))
    state.recent.clear()
    state.recent.extend(pre_trimmed)

    config = {"configurable": {"session_id": session_id}}
    result = chat.invoke({"input": text, "summary": state.summary}, config=config)

    post_trimmed = trim_recent(list(state.recent))
    state.recent.clear()
    state.recent.extend(post_trimmed)

    state.turn_count += 1
    if state.turn_count % MEMORY.summarize_every_n_turns == 0:
        state.summary = update_summary(state.summary, list(state.recent))

    return result.content


if __name__ == "__main__":
    session_id = input("Session ID (default: chat1): ").strip() or "chat1"
    print(f"\nUsing session: {session_id}")
    print(f"Recent window: last {MEMORY.max_turns} turns")
    print(
        f"Token trim recent: {'ON' if MEMORY.enable_token_trimming else 'OFF'} "
        f"(max_tokens_recent={MEMORY.max_tokens_recent})"
    )
    print(f"Summarize every {MEMORY.summarize_every_n_turns} turns")
    print(f"LRU sessions cap: {MEMORY.max_sessions_in_memory}")
    print("Commands: /summary, /sessions, /delete <id>, exit\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        if user_text == "/sessions":
            print("Active sessions (oldest -> newest):")
            print(store.debug_sessions(), "\n")
            continue

        if user_text == "/summary":
            current_summary = store.get(session_id).summary
            print("CURRENT SUMMARY:\n", current_summary or "(empty)", "\n")
            continue

        if user_text.startswith("/delete "):
            session_to_delete = user_text.split(maxsplit=1)[1].strip()
            store.delete(session_to_delete)
            print("Deleted.\n")
            continue

        print("Bot:", chat_once(session_id, user_text), "\n")
