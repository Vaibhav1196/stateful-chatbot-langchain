# Step 3: Session Eviction with LRU

## Goal
Bound total memory footprint by limiting how many sessions stay in memory.

## Theory
Even with bounded per-session memory, unlimited sessions still consume unlimited RAM. LRU eviction solves this.

## Code Explanation
- Adds `SessionHistoryStore` abstraction.
- Implements `InMemoryLRUHistoryStore` with `OrderedDict`.
- Promotes accessed sessions to MRU end.
- Evicts oldest session when capacity is exceeded.
- Keeps per-session bounded history from Step 2.

## Memory Management
- Layer 1: bounded history per session (`deque`).
- Layer 2: bounded number of sessions (LRU).

## DS&A / Python Concept Used
- `OrderedDict` for LRU ordering.
- `threading.RLock` for thread-safe mutation.

## Run This Step
```bash
uv run python steps/step3/stateful_chatbot_step3.py
```

## Expected Behavior
- `/sessions` shows active sessions in LRU order.
- Exceeding capacity evicts oldest inactive sessions.
- `/delete <session_id>` removes a session manually.

## Limitations
- Still in-memory only.
- Multi-process or distributed deployments need Redis/DB.

## What you should test (important)

### 3.1) LRU eviction works

Set:

```python
max_sessions_in_memory=3
```

Then create 5 sessions (`chat1`…`chat5`) and call `/sessions` to see only the most recent 3 remain.

### 3.2) Session isolation still holds

`chat1`: “my name is Amina”

`chat2`: “what’s my name?” → should not know

### 3.3) Memory remains bounded

- Per session: last `max_turns`
- Across sessions: last `max_sessions_in_memory`