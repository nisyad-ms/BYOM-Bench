"""Local Foundry Memory Agent for MemoryGym.

Uses in-process LanceDB memory store that replicates the Azure AI Foundry
Memory pipeline (extraction, consolidation, hybrid search) without any
Foundry API calls. Same tool-calling interface as FoundryMemoryAgent.
"""

import os
import shutil
import threading

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from memory_gym.client import CONFIG, PooledLLMClient, _discover_all_endpoints, get_agent_config
from memory_gym.schemas import MultiSessionOutput

from ._foundry_local_memory import FoundryLocalMemory
from ._tool_calling import respond_with_memory_search

_foundry_local_cfg = get_agent_config("foundry_local")


class FoundryLocalAgent:
    """Agent using local LanceDB-based memory that replicates the Foundry Memory pipeline.

    No Azure Foundry API calls needed. Memory extraction, consolidation, and search
    all run in-process using LanceDB + Azure OpenAI for LLM/embedding operations.

    Interface matches FoundryMemoryAgent: tool-calling loop where the LLM calls
    search_user_memories to retrieve relevant memories.
    """

    def __init__(
        self,
        db_path: str = ".lancedb/foundry_local",
        user_id: str = "default-user",
        completion_model: str | None = None,
        embedding_model: str | None = None,
        num_memories: int | None = None,
    ):
        # Discover endpoint + deployments via multi-endpoint discovery
        pairs = _discover_all_endpoints("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENTS")
        if not pairs:
            raise ValueError("No AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENTS configured")
        endpoint = pairs[0][0]

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", CONFIG["defaults"]["api_version"])

        self._azure_client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

        # Resolve models from discovered pairs
        self.completion_model = completion_model or pairs[0][1]

        embedding_pairs = _discover_all_endpoints("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS")
        if not embedding_pairs:
            raise ValueError("No AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS configured")
        self.embedding_model = embedding_model or embedding_pairs[0][1]

        self.db_path = db_path
        self.user_id = user_id
        self.num_memories = num_memories if num_memories is not None else _foundry_local_cfg["num_memories"]

        self._memory: FoundryLocalMemory | None = None
        self._memory_populated = False
        self._init_lock = threading.Lock()
        self._llm_client: PooledLLMClient | None = None

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Populate local LanceDB memory store from conversation history.

        Thread-safe: only one thread will populate the store.
        """
        with self._init_lock:
            if self._memory_populated:
                return "Local Foundry agent with memory (already populated)"

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

            self._memory_populated = True

        if self._llm_client is None:
            self._llm_client = PooledLLMClient()

        return "Local Foundry agent with memory"

    def _search_memories(self, query: str) -> list[dict[str, str]]:
        """Search local LanceDB memories."""
        if self._memory is None:
            return []
        memories = self._memory.retrieve(self.user_id, query, top_k=self.num_memories)
        return [{"fact": mem} for mem in memories]

    def respond(self, conversation: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Generate a response using Azure OpenAI with local memory search via tool-calling."""
        if self._llm_client is None:
            raise ValueError("LLM client not initialized. Call build_context first.")
        return respond_with_memory_search(
            self._llm_client, "agents/agent_system_memory", conversation, self._search_memories
        )

    def cleanup(self) -> None:
        """Delete the local LanceDB memory store."""
        if self._memory is not None:
            self._memory.cleanup()
            self._memory = None
        self._memory_populated = False

    def reset_conversation(self) -> None:
        """No-op: no per-conversation state to reset."""
