# Bring Your Own Memory Store

This guide explains how to add a custom memory backend to MemoryGym. All you need is a class that implements three methods: `populate`, `retrieve`, and `cleanup`.

## The MemoryStore Protocol

Your store must satisfy the `MemoryStore` protocol defined in `memory_gym/agents/stores/protocol.py`:

```python
class MemoryStore(Protocol):
    def populate(self, multisession_data: MultiSessionOutput) -> None: ...
    def retrieve(self, query: str) -> list[str]: ...
    def cleanup(self) -> None: ...
```

| Method | Called | Purpose |
|--------|--------|---------|
| `populate` | Once per evaluation session | Ingest conversation history into your store |
| `retrieve` | Once per agent turn during evaluation | Return relevant facts for a search query |
| `cleanup` | After all tasks for a session complete | Delete resources (files, cloud state, etc.) |

You don't need to inherit from anything. Python's structural typing means any class with these three methods works.

## Data Flow

```
MultiSessionOutput
  └── sessions: list[Session]
        └── conversation: list[dict]  # [{"role": "user", "content": "..."}, ...]
```

During evaluation:
1. `MemoryAgent` calls `store.populate(multisession_data)` once
2. For each agent turn, the tool-calling loop calls `store.retrieve(query)` with a natural language query
3. Returned strings are injected into the LLM context as `{"fact": "..."}` dicts
4. After all tasks complete, `MemoryAgent` calls `store.cleanup()`

## Minimal Example

> **Note:** This example is for understanding the protocol only. It won't be autodiscovered by `--agent` because it lacks `_sentinel_agent_type`. See the [full example](#full-example-with-config-and-sentinels) below for a store that plugs into the evaluation harness automatically.

```python
# memory_gym/agents/stores/my_store.py

from memory_gym.schemas import MultiSessionOutput


class MyMemoryStore:
    """Minimal in-memory store using substring matching."""

    def __init__(self):
        self._facts: list[str] = []

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        for session in multisession_data.sessions:
            for msg in session.conversation:
                if msg["role"] == "user":
                    self._facts.append(msg["content"])

    def retrieve(self, query: str) -> list[str]:
        query_lower = query.lower()
        return [f for f in self._facts if query_lower in f.lower()][:5]

    def cleanup(self) -> None:
        self._facts.clear()
```

Use it:

```python
from memory_gym.agents import MemoryAgent
from memory_gym.agents.stores.my_store import MyMemoryStore

store = MyMemoryStore()
agent = MemoryAgent(store)
agent.build_context(multisession_data)          # calls store.populate()
response, memories = agent.respond(conversation) # calls store.retrieve() per turn
agent.cleanup()                                  # calls store.cleanup()
```

## Full Example with Config and Sentinels

A sentinel is a small JSON file that records how many sessions have been ingested into a store. When an evaluation run is interrupted and resumed (via `--reuse-stores`), the store reads its sentinel to decide whether it can skip re-population. If the sentinel exists and the session count matches, the store reuses the existing data instead of rebuilding from scratch. Sentinels are written after a successful `populate()` and deleted on `cleanup()`.

Sentinel files are written to `.memory_sentinels/` (created automatically when `--reuse-stores` is passed). The filename follows the pattern `{session_name}_{agent_type}.json`, e.g. `.memory_sentinels/2026-02-02_1414_mem0.json`. When `--reuse-stores` is not passed, `sentinel_dir` is `None` and no sentinel files are written — the store rebuilds every run.

For a production store that supports `--reuse-stores`, follow this pattern. The autodiscovery registry automatically scans for classes ending in `MemoryStore` that have a `_sentinel_agent_type` class variable — your store will appear in `--agent` choices and smoke tests without any manual wiring. See `mem0.py` or `foundry_local.py` for real implementations.

### 1. Create the store

```python
# memory_gym/agents/stores/my_store.py

import os
import shutil
from pathlib import Path

from memory_gym.client import get_agent_config
from memory_gym.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

MEMORY_STORE_NAME = "my_store"

_cfg = get_agent_config(MEMORY_STORE_NAME)


class MyMemoryStore(SentinelMixin):
    _sentinel_agent_type = MEMORY_STORE_NAME   # Unique key — used for sentinel filenames and autodiscovery.
                                               # Must match your config filename (configs/agents/<name>.yaml).

    def __init__(
        self,
        *,
        session_dir: Path,                          # Session output dir (e.g. outputs/2026-02-02_1414).
                                                    # Provided automatically by the evaluation harness.
                                                    # Use session_dir.name to derive unique local paths.
        num_memories: int | None = None,            # Number of memories returned by retrieve(). Falls
                                                    # back to configs/agents/<name>.yaml (default: 5).
        sentinel_dir: Path | None = None,           # Where to write sentinel files. None = no sentinels
                                                    # (store rebuilds every run). Set by --reuse-stores.
        session_name: str | None = None,            # Human-readable session label used in sentinel
                                                    # filenames. Typically session_dir.name.
    ):
        self.db_path = f".{MEMORY_STORE_NAME}/{session_dir.name}"
        self.num_memories = num_memories if num_memories is not None else _cfg["num_memories"]
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

        # Initialize your backend here
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # self._client = ...

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        # Check sentinel — skip rebuild if store is already populated
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            if os.path.isdir(self.db_path):
                print(f"Reusing existing store at {self.db_path} (sentinel valid)")
                return
        self._delete_sentinel()

        # Remove stale data
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        # Ingest each session's conversation chronologically
        for session in multisession_data.sessions:
            if not session.conversation:
                continue
            messages = [{"role": m["role"], "content": m["content"]} for m in session.conversation]
            # ... your ingestion logic here ...

        self._write_sentinel(len(multisession_data.sessions))

    def retrieve(self, query: str) -> list[str]:
        # Return up to self.num_memories fact strings
        # ... your retrieval logic here ...
        return []

    def cleanup(self) -> None:
        self._delete_sentinel()
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
```

### 2. Create a config file

```yaml
# configs/agents/my_store.yaml
num_memories: 5
```

### 3. Add to `.gitignore`

If your store writes to a local directory (e.g. `.my_store/`), add it to `.gitignore`.

## What MemoryAgent Gives You

When you wrap your store in `MemoryAgent`, you get for free:

- **Thread-safe populate**: `build_context()` uses a lock so concurrent eval tasks don't double-populate
- **PooledLLMClient**: Multi-endpoint Azure OpenAI client with load balancing
- **Tool-calling loop**: The agent LLM decides when to search memory via function calling; your `retrieve()` is called automatically
- **Conversation management**: System prompts, turn tracking, response extraction

## Design Notes

- **Chronological ingestion**: Sessions are ordered. Ingest them in order if your backend is order-sensitive.
- **`retrieve()` returns plain strings**: Each string should be a self-contained fact (e.g., `"Prefers dark mode in all applications"`). The `MemoryAgent` wraps them as `{"fact": text}`.
- **Cleanup must be thorough**: The evaluation harness creates a fresh store per session. Leftover state from previous runs causes incorrect results.
- **Sentinels are optional**: Only needed if you want `--reuse-stores` support. Without them, the store rebuilds every run.
- **No ABC inheritance needed**: Just implement the three methods. Python's `Protocol` handles the rest.

## Existing Stores as Reference

| Store | File | Backend | Good reference for |
|-------|------|---------|--------------------|
| `FoundryLocalMemoryStore` | `stores/foundry_local.py` | LanceDB | Local DB + sentinel pattern |
| `Mem0MemoryStore` | `stores/mem0.py` | Qdrant + mem0 | Third-party library integration |
| `Mem0GraphMemoryStore` | `stores/mem0_graph.py` | Qdrant + Kùzu + mem0 | Vector + graph hybrid retrieval |
| `AWSMemoryStore` | `stores/aws.py` | Bedrock AgentCore | Cloud API with polling and retry |
| `ZepMemoryStore` | `stores/zep.py` | Graphiti + Kùzu | Graph-based knowledge extraction |
| `HindsightMemoryStore` | `stores/hindsight.py` | Hindsight | Auto-detecting embedding dimensions |
