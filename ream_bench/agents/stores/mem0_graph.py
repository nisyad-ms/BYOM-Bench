"""Mem0 Graph Memory Store for BYOM-Bench.

Uses the mem0 library with Kùzu graph database for entity/relationship extraction
and graph-based retrieval alongside vector search. Backed by local Qdrant for
vector storage, Kùzu for graph storage, and Azure OpenAI for LLM/embedding operations.
"""

import os
import shutil
from pathlib import Path

from mem0 import Memory

from byom_bench.client import get_agent_config, resolve_azure_openai_config
from byom_bench.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

_mem0_graph_cfg = get_agent_config("mem0_graph")


class Mem0GraphMemoryStore(SentinelMixin):
    """Memory store backed by mem0 with Kùzu graph (vector + knowledge graph retrieval).

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    Uses Azure OpenAI for LLM/embedding (via DefaultAzureCredential when no API key
    is set), local Qdrant for vector storage, and Kùzu for graph storage.
    """

    _sentinel_agent_type = "mem0_graph"

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

        db_path = f".qdrant_mem0_graph/{session_dir.name}"
        graph_db_path = f".kuzu_mem0/{session_dir.name}"

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(graph_db_path), exist_ok=True)

        # Build mem0 config with graph store
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
                    "embedding_dims": 1536,
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
                    "collection_name": "mem0_graph_memories",
                    "path": db_path,
                },
            },
            "graph_store": {
                "provider": "kuzu",
                "config": {
                    "db": graph_db_path,
                },
            },
            "history_db_path": os.path.join(os.path.dirname(db_path), "mem0_graph_history.db"),
        }

        self._memory = Memory.from_config(config)
        self.user_id = user_id
        self.db_path = db_path
        self.graph_db_path = graph_db_path
        self._history_db_path = config["history_db_path"]
        self.num_memories = num_memories if num_memories is not None else _mem0_graph_cfg["num_memories"]
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
                print(f"Reusing existing mem0_graph store at {self.db_path} (sentinel valid)")
                return
        self._delete_sentinel()

        # Delete stale stores before populating
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Removed stale mem0_graph vector store at {self.db_path}")
        if os.path.exists(self.graph_db_path):
            if os.path.isdir(self.graph_db_path):
                shutil.rmtree(self.graph_db_path)
            else:
                os.remove(self.graph_db_path)
            print(f"Removed stale mem0_graph graph store at {self.graph_db_path}")

        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            messages = [{"role": msg["role"], "content": msg["content"]} for msg in session.conversation]
            result = self._memory.add(messages, user_id=self.user_id)
            num_ops = len(result.get("results", []))
            print(f"Session {session.session_id}: {num_ops} memory operations (mem0_graph)", flush=True)

        self._write_sentinel(
            len(multisession_data.sessions), store_path=self.db_path, graph_store_path=self.graph_db_path
        )

    def retrieve(self, query: str) -> list[str]:
        """Search mem0 memories (vector + graph) and return fact strings."""
        results = self._memory.search(query, user_id=self.user_id, limit=self.num_memories)

        memories: list[str] = []

        # Vector memories
        for r in results.get("results", []):
            memories.append(r["memory"])

        # Graph relations (triplets)
        for rel in results.get("relations", []):
            memories.append(f"{rel['source']} - {rel['relationship']} - {rel['destination']}")

        return memories

    def cleanup(self) -> None:
        """Delete all mem0 memories for this user and remove local stores."""
        self._delete_sentinel()
        try:
            self._memory.delete_all(user_id=self.user_id)
        except Exception:
            pass  # Best-effort cleanup
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        if os.path.exists(self.graph_db_path):
            if os.path.isdir(self.graph_db_path):
                shutil.rmtree(self.graph_db_path)
            else:
                os.remove(self.graph_db_path)
            os.remove(self._history_db_path)
