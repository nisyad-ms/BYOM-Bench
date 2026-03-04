"""Mem0 Memory Store for MemoryGym.

Uses the mem0 library (https://github.com/mem0ai/mem0) for automatic memory
extraction, consolidation, and semantic search. Backed by local Qdrant for
vector storage and Azure OpenAI for LLM/embedding operations.
"""

import os
import shutil
from pathlib import Path

from mem0 import Memory

from memory_gym.client import get_agent_config, resolve_azure_openai_config
from memory_gym.schemas import MultiSessionOutput

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

        db_path = f".qdrant/{session_dir.name}"

        # Build mem0 config — omitting api_key triggers DefaultAzureCredential
        config = {
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
                    "path": db_path,
                },
            },
            "history_db_path": os.path.join(os.path.dirname(db_path), "mem0_history.db"),
        }

        # Ensure parent directory exists for Qdrant and history DB
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._memory = Memory.from_config(config)
        self.user_id = user_id
        self.db_path = db_path
        self._history_db_path = config["history_db_path"]
        self.num_memories = num_memories if num_memories is not None else _mem0_cfg["num_memories"]
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

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
                print(f"Reusing existing mem0 store at {self.db_path} (sentinel valid)")
                return
        self._delete_sentinel()

        # Delete stale store before populating
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Removed stale mem0 store at {self.db_path}")

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
        results = self._memory.search(query, user_id=self.user_id, limit=self.num_memories)
        return [r["memory"] for r in results.get("results", [])]

    def cleanup(self) -> None:
        """Delete all mem0 memories for this user and remove local store."""
        self._delete_sentinel()
        try:
            self._memory.delete_all(user_id=self.user_id)
        except Exception:
            pass  # Best-effort cleanup
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        if os.path.exists(self._history_db_path):
            os.remove(self._history_db_path)
