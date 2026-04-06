"""Mem0 Memory Store for REAM-Bench.

Uses the mem0 library (https://github.com/mem0ai/mem0) for automatic memory
extraction, consolidation, and semantic search. Backed by local Qdrant for
vector storage and Azure OpenAI for LLM/embedding operations.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import patch

from mem0 import Memory

from ream_bench.client import get_agent_config, resolve_azure_openai_config
from ream_bench.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

_mem0_cfg = get_agent_config("mem0")


class Mem0MemoryStore(SentinelMixin):
    """Memory store backed by mem0 (automatic extraction + Qdrant vector search).

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    Uses Azure OpenAI for LLM/embedding (via DefaultAzureCredential when no API key
    is set) and local Qdrant for vector storage.
    """

    _sentinel_agent_type = "mem0"

    def __init__(
        self,
        *,
        session_dir: Path,
        user_id: str = "default-user",
        num_memories: int | None = None,
        sentinel_dir: Path | None = None,
        session_name: str | None = None,
    ):
        # Discover Azure OpenAI endpoints
        endpoint, chat_deployment, emb_endpoint, emb_deployment, api_version = resolve_azure_openai_config()

        self.db_path = f".qdrant/{session_dir.name}"
        self.user_id = user_id
        self.num_memories = num_memories if num_memories is not None else _mem0_cfg["num_memories"]
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

        # Store config for lazy Memory initialization
        self._mem0_config = {
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": chat_deployment,
                    "temperature": 0,
                    "azure_kwargs": {
                        "azure_deployment": chat_deployment,
                        "azure_endpoint": endpoint,
                        "api_version": api_version,
                    },
                },
            },
            "embedder": {
                "provider": "azure_openai",
                "config": {
                    "model": emb_deployment,
                    "azure_kwargs": {
                        "azure_deployment": emb_deployment,
                        "azure_endpoint": emb_endpoint,
                        "api_version": api_version,
                    },
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_memories",
                    "path": self.db_path,
                },
            },
            "history_db_path": os.path.join(self.db_path, "mem0_history.db"),
        }
        self._history_db_path = self._mem0_config["history_db_path"]
        self._memory: Memory | None = None

    def _init_memory(self) -> Memory:
        """Create (or recreate) the mem0 Memory object.

        Patches out mem0's telemetry vector store creation to avoid concurrent
        sessions fighting over the shared ``~/.mem0/migrations_qdrant/.lock``.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # mem0 creates a shared Qdrant instance at ~/.mem0/migrations_qdrant/
        # for telemetry. Its exclusive file lock prevents concurrency.
        from mem0.utils.factory import VectorStoreFactory

        _real_create = VectorStoreFactory.create

        @staticmethod
        def _patched_create(provider, config):
            if getattr(config, "collection_name", None) == "mem0migrations":
                return None
            return _real_create(provider, config)

        with patch.object(VectorStoreFactory, "create", _patched_create), \
             patch("mem0.memory.main.capture_event"):
            self._memory = Memory.from_config(self._mem0_config)
        return self._memory

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Ingest all sessions into mem0 via its automatic extraction pipeline.

        If a valid sentinel exists and the Qdrant store is on disk, reuses it.
        """
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            if os.path.isdir(self.db_path):
                self._init_memory()
                print(f"Reusing existing mem0 store at {self.db_path} (sentinel valid)")
                return
        self._delete_sentinel()

        # Delete stale store BEFORE creating Memory object
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Removed stale mem0 store at {self.db_path}")

        # Now create Memory with a clean directory
        self._init_memory()

        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            messages = [{"role": msg["role"], "content": msg["content"]} for msg in session.conversation]
            result = self._memory.add(messages, user_id=self.user_id)
            num_ops = len(result.get("results", []))
            print(f"Session {session.session_id}: {num_ops} memory operations (mem0)", flush=True)

        self._write_sentinel(len(multisession_data.sessions), store_path=self.db_path)

    def retrieve(self, query: str) -> list[str]:
        """Search mem0 memories and return fact strings."""
        if self._memory is None:
            return []
        results = self._memory.search(query, user_id=self.user_id, limit=self.num_memories)
        return [r["memory"] for r in results.get("results", [])]

    def cleanup(self) -> None:
        """Delete all mem0 memories for this user and remove local store."""
        self._delete_sentinel()
        if self._memory is not None:
            try:
                self._memory.delete_all(user_id=self.user_id)
            except Exception:
                pass  # Best-effort cleanup
        self._memory = None
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        if os.path.exists(self._history_db_path):
            os.remove(self._history_db_path)
