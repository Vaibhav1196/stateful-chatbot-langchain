# Memory Management Progression

## Step 1: Basic In-Memory Session State
- Data structure: Python `dict`
- Mapping: `session_id -> ChatMessageHistory`
- Benefit: O(1) average lookup/insert
- Limitation: unbounded growth and no persistence

## Step 2: Bounded Memory
Two controls are combined:
1. Window memory (`deque(maxlen=2 * max_turns)`) to cap message count.
2. Token trimming (`trim_messages`) to cap context size in tokens.

Result:
- bounded per-session memory
- lower risk of context overflow
- controlled latency/cost behavior

## Step 3: LRU Session Eviction + Store Abstraction
Even bounded sessions can accumulate endlessly. Step 3 caps total sessions with an LRU cache (`OrderedDict`).

Behavior:
- access promotes session to most-recently-used
- overflow triggers oldest-session eviction
- lock-based synchronization protects concurrent updates

## Step 4: Prompt Discipline + Summary Memory
Step 4 introduces hierarchical memory:
- **Recent memory**: short-term transcript window (`deque`).
- **Summary memory**: compressed durable context (periodically updated).

Why this helps:
- old but important facts survive trimming
- context stays compact
- conversation coherence improves for longer sessions

## DS&A Concepts Used Across Steps
- `dict` (hash map)
- `deque` (bounded queue, O(1) append/pop at ends)
- `OrderedDict` (LRU ordering)
- lock-protected critical sections for thread safety
