"""Generic memory-backed agent that wraps any MemoryStore.

Provides the shared plumbing that all memory agents need:
- Thread-safe ``build_context`` (populate once, reuse across eval runs)
- ``PooledLLMClient`` lifecycle
- Tool-calling loop via ``respond_with_memory_search``
- ``cleanup`` delegation to the underlying store
"""

import threading

from ream_bench.client import PooledLLMClient
from ream_bench.schemas import MultiSessionOutput

from ._internal.tool_calling import respond_with_memory_search
from .stores.protocol import MemoryStore


class MemoryAgent:
    """Wraps a ``MemoryStore`` and drives the Azure OpenAI tool-calling loop.

    Usage::

        store = AWSMemoryStore(memory_name="my-session", ...)
        agent = MemoryAgent(store)
        agent.build_context(multisession_data)
        text, memories = agent.respond(conversation)
        agent.cleanup()
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._store = memory_store
        self._llm_client: PooledLLMClient | None = None
        self._memory_populated = False
        self._init_lock = threading.Lock()

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Populate the store and initialise the LLM pool (thread-safe)."""
        with self._init_lock:
            if self._memory_populated:
                return "Memory agent (already populated)"
            self._store.populate(multisession_data)
            self._memory_populated = True
        if self._llm_client is None:
            self._llm_client = PooledLLMClient()
        return "Memory agent with search"

    def _search_memories(self, query: str) -> list[dict[str, str]]:
        """Adapt ``store.retrieve`` to the dict format the tool-loop expects."""
        facts = self._store.retrieve(query)
        return [{"fact": f} for f in facts]

    def respond(self, conversation: list[dict[str, str]], memory_token_budget: int | None = None) -> tuple[str, list[dict]]:
        """Run the tool-calling loop with memory search."""
        if self._llm_client is None:
            raise ValueError("LLM client not initialised. Call build_context first.")
        return respond_with_memory_search(
            self._llm_client, "agents/agent_system_memory", conversation, self._search_memories, memory_token_budget
        )

    def reset_conversation(self) -> None:
        """No-op: no per-conversation state to reset."""

    def cleanup(self) -> None:
        """Forward to the underlying store."""
        self._store.cleanup()
        self._memory_populated = False
