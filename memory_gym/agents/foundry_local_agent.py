"""Local Foundry Memory Agent for MemoryGym.

Uses in-process LanceDB memory store that replicates the Azure AI Foundry
Memory pipeline (extraction, consolidation, hybrid search) without any
Foundry API calls. Same tool-calling interface as FoundryMemoryAgent.
"""

import os
import threading

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from memory_gym.client import CONFIG, PooledLLMClient, _parse_env_list, get_agent_config
from memory_gym.schemas import MultiSessionOutput

from ._foundry_local_memory import FoundryLocalMemory
from ._tool_calling import respond_with_memory_search

_foundry_local_cfg = get_agent_config("foundry_local")


def _get_foundry_embedding_deployments() -> list[str]:
    return _parse_env_list("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS")


class FoundryLocalAgent:
    """Agent using local LanceDB-based memory that replicates the Foundry Memory pipeline.

    No Azure Foundry API calls needed. Memory extraction, consolidation, and search
    all run in-process using LanceDB + Azure OpenAI for LLM/embedding operations.

    Interface matches FoundryMemoryAgent: tool-calling loop where the LLM calls
    search_user_memories to retrieve relevant memories.
    """

    def __init__(
        self,
        db_path: str = "./.lancedb_foundry_local",
        user_id: str = "default-user",
        completion_model: str | None = None,
        embedding_model: str | None = None,
        num_memories: int | None = None,
    ):
        # Azure OpenAI client for extraction/consolidation/embedding
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set in environment")

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", CONFIG["defaults"]["api_version"])

        self._azure_client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

        # Resolve models — require env vars, no fallbacks
        deployments = _parse_env_list("AZURE_OPENAI_DEPLOYMENTS")
        if not deployments:
            raise ValueError("AZURE_OPENAI_DEPLOYMENTS not set in environment")
        self.completion_model = completion_model or deployments[0]

        embedding_deployments = _get_foundry_embedding_deployments()
        if not embedding_deployments:
            raise ValueError("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS not set in environment")
        self.embedding_model = embedding_model or embedding_deployments[0]

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
                import shutil

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
