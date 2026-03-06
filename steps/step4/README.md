# Step 4: Prompt Discipline + Summary Memory

## Goal
Keep long conversations coherent while staying within bounded context.

## Theory
Window trimming can discard important old facts. Step 4 introduces two-level memory:
- recent transcript window (short-term)
- summary memory (compressed long-term)

## Code Explanation
- Adds `SessionState(summary, recent, turn_count)`.
- Prompt includes system rules, summary, recent history, then current input.
- Every N turns, summary is refreshed using model-based summarization.
- Sensitive data retention is discouraged via explicit summarization rules.

## Memory Management
- Recent memory: bounded deque + optional token trim.
- Summary memory: compact durable facts/preferences/goals.
- Session-level LRU still bounds total active sessions.

## DS&A / Python Concept Used
- Composite session state (`dataclass`) per `session_id`.
- LRU `OrderedDict` + bounded `deque`.
- Periodic compression strategy for long-term memory.

## Run This Step
```bash
uv run python steps/step4/stateful_chatbot_step4.py
```

## Expected Behavior
- Recent chat remains responsive and bounded.
- `/summary` shows compressed long-term memory.
- Important stable facts can survive recent-window trimming.

## Limitations
- Summary quality depends on model behavior.
- No external persistence or governance layer yet.


## What to test (to prove it works)

### Test A: “Summary survives trimming”

1. Set `max_turns=2` (small window)
2. Chat 10+ turns and introduce a stable fact early:
    
    “My name is Amina. I’m building a LangChain project.”
    
3. After many turns, ask: “What’s my name and what am I building?”
4. Check `/summary` to see the compressed memory.

### Test B: “Summary updates periodically”

Chat until you hit `summarize_every_n_turns` (default 6).

Then run `/summary` — it should no longer be empty.
