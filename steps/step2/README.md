# Step 2: Bounded Memory

## Goal
Prevent history from growing forever by limiting both message count and token budget.

## Theory
Two risks appear in Step 1: unbounded memory and context-window overflow. Step 2 introduces policy-based memory control.

## Code Explanation
- Uses `deque(maxlen=2 * max_turns)` as bounded message window.
- Uses `trim_messages()` for token-based trimming.
- Applies trimming before and after each model call.

## Memory Management
- Window memory: keeps only recent K turns.
- Token trimming: removes older content to stay under max token budget.

## DS&A / Python Concept Used
- `deque`: bounded queue with O(1) append and automatic eviction from the left.

## Run This Step
```bash
uv run python steps/step2/stateful_chatbot_step2.py
```

## Expected Behavior
- Older messages are evicted automatically once the window limit is reached.
- Large context is trimmed to token budget.

## Limitations
- Total number of sessions is still unbounded.
- Memory is still process-local and non-persistent.

## Mini-tests (do these now)

1. Set `max_turns=2`, chat for 8–10 turns, then ask:
    - “What did I say at the very beginning?”
    - It should forget early content.
2. Paste a long paragraph in one message.
    - With token trimming enabled, it should still keep the most recent context and not explode.
3. Confirm multi-session isolation still works (`chat1`, `chat2`).