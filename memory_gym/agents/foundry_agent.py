"""Foundry Memory Agent using Azure AI Foundry memory store."""

import os
import threading
import uuid

import openai
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
    ResponsesAssistantMessageItemParam,
    ResponsesSystemMessageItemParam,
    ResponsesUserMessageItemParam,
)
from azure.identity import DefaultAzureCredential
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory_gym.schemas import MultiSessionOutput


def _get_foundry_deployments() -> list[str]:
    """Get Foundry deployment names from AZURE_FOUNDRY_DEPLOYMENTS env var."""
    deployments_str = os.environ.get("AZURE_FOUNDRY_DEPLOYMENTS", "")
    if not deployments_str:
        return []
    return [d.strip() for d in deployments_str.split(",") if d.strip()]


def _get_foundry_embedding_deployments() -> list[str]:
    deployments_str = os.environ.get("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS", "")
    if not deployments_str:
        return []
    return [d.strip() for d in deployments_str.split(",") if d.strip()]


def get_foundry_embedding_models() -> list[str]:
    return _get_foundry_embedding_deployments() or ["text-embedding-3-small-001"]


class FoundryMemoryAgent:
    """Agent using Azure AI Foundry memory store for preference recall.

    Creates one Foundry agent per deployment model and routes respond() calls
    to the least-busy agent, spreading load across all endpoints.
    """

    def __init__(
        self,
        memory_store_name: str,
        scope: str = "user",
        chat_models: list[str] | None = None,
        embedding_model: str | None = None,
    ):
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_FOUNDRY_ENDPOINT not set in environment")

        self.client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        self.memory_store_name = memory_store_name
        self.scope = scope
        if embedding_model is None:
            embedding_model = (_get_foundry_embedding_deployments() or ["text-embedding-3-small-001"])[0]
        self.embedding_model = embedding_model
        self._memory_populated = False
        self._init_lock = threading.Lock()

        if chat_models is None:
            chat_models = _get_foundry_deployments() or ["gpt-4.1-001"]
        self.chat_models = chat_models
        self._agents: list = []
        self._agent_ids = [str(uuid.uuid4())[:8] for _ in chat_models]
        self._in_flight = [0] * len(chat_models)
        self._local = threading.local()

    def ensure_memory_store(self, multisession_data: MultiSessionOutput, force_recreate: bool = False) -> None:
        """Create memory store if needed and populate with conversation history.

        Thread-safe: only one thread will create/populate the store.
        """
        with self._init_lock:
            if self._memory_populated and not force_recreate:
                return

            existing_stores = [ms.name for ms in self.client.memory_stores.list()]

            if self.memory_store_name in existing_stores:
                if force_recreate:
                    self.client.memory_stores.delete(self.memory_store_name)
                else:
                    self._memory_populated = True
                    return

            definition = MemoryStoreDefaultDefinition(
                chat_model=self.chat_models[0],
                embedding_model=self.embedding_model,
                options=MemoryStoreDefaultOptions(
                    user_profile_enabled=True,
                    chat_summary_enabled=True,
                ),
            )

            self.client.memory_stores.create(
                name=self.memory_store_name,
                definition=definition,
                description=f"Memory store for persona: {multisession_data.persona[:50]}",
            )

            self._populate_memories(multisession_data)
            self._memory_populated = True

    def _populate_memories(self, multisession_data: MultiSessionOutput) -> None:
        update_poller = None
        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            messages = _to_foundry_messages(session.conversation)
            if not messages:
                continue

            update_poller = self.client.memory_stores.begin_update_memories(
                name=self.memory_store_name,
                scope=self.scope,
                items=messages,
                previous_update_id=update_poller.update_id if update_poller else None,
                update_delay=0,
            )

            result = update_poller.result()
            print(f"Session {session.session_id}: {len(result.memory_operations)} memory operations")

    def _ensure_agents(self) -> None:
        """Create one Foundry agent per deployment model.

        Thread-safe: only one thread will create the agents.
        """
        with self._init_lock:
            if self._agents:
                return

            tool = MemorySearchTool(
                memory_store_name=self.memory_store_name,
                scope=self.scope,
                update_delay=300,
            )

            for i, model in enumerate(self.chat_models):
                agent = self.client.agents.create_version(
                    agent_name=f"PersonaGymAgent-{self._agent_ids[i]}",
                    definition=PromptAgentDefinition(
                        model=model,
                        instructions="You are a helpful assistant that remembers user preferences and applies them proactively.",
                        tools=[tool],
                    ),
                )
                self._agents.append(agent)

    def _acquire_agent(self) -> tuple[int, object]:
        with self._init_lock:
            idx = min(range(len(self._in_flight)), key=lambda i: self._in_flight[i])
            self._in_flight[idx] += 1
            return idx, self._agents[idx]

    def _release_agent(self, idx: int) -> None:
        with self._init_lock:
            self._in_flight[idx] -= 1

    def build_context(self, multisession_data: MultiSessionOutput, force_recreate: bool = False) -> str:
        """Ensure memory store is populated and agents are ready."""
        self.ensure_memory_store(multisession_data, force_recreate=force_recreate)
        self._ensure_agents()
        return "Foundry agent with memory search"

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response using least-busy Foundry agent.

        Thread-safe: each thread gets its own conversation ID per agent index.
        """
        if not self._agents:
            raise ValueError("Agents not initialized. Call build_context first.")

        idx, agent = self._acquire_agent()
        try:
            conv_key = f"conversation_id_{idx}"
            openai_client = self.client.get_openai_client()

            conv_id = getattr(self._local, conv_key, None)
            if conv_id is None:
                conv = openai_client.conversations.create()
                setattr(self._local, conv_key, conv.id)
                conv_id = conv.id

            last_user_message = ""
            for msg in reversed(conversation):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break

            response = self._call_with_retry(
                openai_client,
                last_user_message,
                conv_id,
                agent,
            )

            return response.output_text
        finally:
            self._release_agent(idx)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(12),
    )
    def _call_with_retry(self, openai_client, message: str, conversation_id: str, agent):
        """Call responses.create with retry logic for rate limits."""
        return openai_client.responses.create(
            input=message,
            conversation=conversation_id,
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
        )

    def reset_conversation(self) -> None:
        """Reset all conversation IDs for the current thread."""
        for i in range(len(self.chat_models)):
            setattr(self._local, f"conversation_id_{i}", None)

    def delete_memory_store(self) -> None:
        """Delete the memory store (for cleanup)."""
        try:
            self.client.memory_stores.delete(self.memory_store_name)
        except Exception:
            pass


def _to_foundry_messages(
    conversation: list[dict[str, str]],
) -> list[ResponsesSystemMessageItemParam | ResponsesUserMessageItemParam | ResponsesAssistantMessageItemParam]:
    """Convert conversation to Foundry message format."""
    messages = []

    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            messages.append(ResponsesSystemMessageItemParam(content=content))
        elif role == "user":
            messages.append(ResponsesUserMessageItemParam(content=content))
        elif role == "assistant":
            messages.append(ResponsesAssistantMessageItemParam(content=content))

    return messages
