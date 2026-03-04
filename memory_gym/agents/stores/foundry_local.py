"""Local Foundry Memory Store for MemoryGym.

Uses in-process LanceDB memory store that replicates the Azure AI Foundry
Memory pipeline (extraction, consolidation, hybrid search) without any
Foundry API calls.
"""

import os
import shutil
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from memory_gym.client import get_agent_config, resolve_azure_openai_config
from memory_gym.schemas import MultiSessionOutput

from .._internal.foundry_local_memory import FoundryLocalMemory
from ._sentinel import SentinelMixin

_foundry_local_cfg = get_agent_config("foundry_local")


class FoundryLocalMemoryStore(SentinelMixin):
    """Memory store backed by local LanceDB (replicates Foundry Memory pipeline).

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    No Azure Foundry API calls needed. Memory extraction, consolidation, and search
    all run in-process using LanceDB + Azure OpenAI for LLM/embedding operations.
    """

    _sentinel_agent_type = "foundry_local"

    def __init__(
        self,
        *,
        session_dir: Path,
        user_id: str = "default-user",
        num_memories: int | None = None,
        sentinel_dir: Path | None = None,
        session_name: str | None = None,
    ):
        # Discover endpoint + deployments via multi-endpoint discovery
        endpoint, chat_deployment, _emb_endpoint, emb_deployment, api_version = resolve_azure_openai_config()

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        self._azure_client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

        self.completion_model = chat_deployment
        self.embedding_model = emb_deployment

        self.db_path = f".lancedb/{session_dir.name}"
        self.user_id = user_id
        self.num_memories = num_memories if num_memories is not None else _foundry_local_cfg["num_memories"]

        self._memory: FoundryLocalMemory | None = None
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Populate local LanceDB memory store from conversation history.

        If a valid sentinel exists and the store is on disk, reuses it.
        """
        # Check sentinel for existing valid store
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            if os.path.isdir(self.db_path):
                self._memory = FoundryLocalMemory(
                    completion_client=self._azure_client,
                    embedding_client=self._azure_client,
                    completion_model=self.completion_model,
                    embedding_model=self.embedding_model,
                    db_path=self.db_path,
                )
                print(f"Reusing existing LanceDB store at {self.db_path} (sentinel valid)")
                return
        # Sentinel invalid or store missing — delete stale sentinel
        self._delete_sentinel()

        # Delete any stale LanceDB directory before populating
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Removed stale LanceDB at {self.db_path}")

        self._memory = FoundryLocalMemory(
            completion_client=self._azure_client,
            embedding_client=self._azure_client,
            completion_model=self.completion_model,
            embedding_model=self.embedding_model,
            db_path=self.db_path,
        )

        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            num_ops = self._memory.add(self.user_id, session.conversation)
            print(f"Session {session.session_id}: {num_ops} memory operations", flush=True)

        self._write_sentinel(len(multisession_data.sessions), store_id=self.db_path)

    def retrieve(self, query: str) -> list[str]:
        """Search local LanceDB memories and return fact strings."""
        if self._memory is None:
            return []
        return self._memory.retrieve(self.user_id, query, top_k=self.num_memories)

    def cleanup(self) -> None:
        """Delete the local LanceDB memory store and sentinel."""
        self._delete_sentinel()
        if self._memory is not None:
            self._memory.cleanup()
            self._memory = None
