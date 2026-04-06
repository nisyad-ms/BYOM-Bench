"""MemoryStore protocol for pluggable memory backends.

Implement this protocol to add a new memory backend to BYOM-Bench.
Wrap your implementation in a ``MemoryAgent`` to get the shared
tool-calling loop, thread-safe init, and PooledLLMClient lifecycle
for free.
"""

from typing import Protocol

from byom_bench.schemas import MultiSessionOutput


class MemoryStore(Protocol):
    """Minimal interface that every memory backend must satisfy.

    Methods:
        populate: Ingest multi-session conversation history into the store.
        retrieve: Return relevant facts for a search query.
        cleanup:  Release cloud / local resources owned by the store.
    """

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Ingest all sessions into the memory store.

        Called exactly once per evaluation session (thread-safe guard is
        in ``MemoryAgent``).  Implementations may create cloud resources,
        send conversation events, and block until extraction completes.
        """
        ...

    def retrieve(self, query: str) -> list[str]:
        """Return fact strings relevant to *query*.

        The generic ``MemoryAgent`` wraps each string into the
        ``{"fact": text}`` dict expected by the tool-calling loop.
        """
        ...

    def cleanup(self) -> None:
        """Delete cloud resources / local files created by this store."""
        ...
