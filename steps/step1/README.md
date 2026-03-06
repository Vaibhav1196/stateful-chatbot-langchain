# Step 1: Basic Stateful Chatbot

## Goal
Build a chatbot where each `session_id` preserves its own conversation history.

## Theory
A chatbot is stateful when past messages from a session are automatically included in later turns.

## Code Explanation
- `store` is a dictionary mapping `session_id` to `ChatMessageHistory`.
- `get_session_history` creates history lazily.
- `RunnableWithMessageHistory` manages loading and appending messages.

## Memory Management
- In-memory only
- Fast lookup
- Lost on restart
- Unbounded growth

## DS&A / Python Concept Used
- `dict` (hash map): average O(1) insert and lookup

## Run This Step
```bash
uv run python steps/step1/stateful_chatbot_step1.py
```

## Expected Behavior
- Reusing the same session ID keeps context across turns.
- Using a new session ID starts a fresh conversation.

## Limitations
- No memory bounds
- No token controls
- No session eviction
