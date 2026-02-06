"""Foundry Memory Agent using Azure AI Foundry memory store."""

import logging
import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchOptions,
    MemorySearchTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
    ResponsesAssistantMessageItemParam,
    ResponsesSystemMessageItemParam,
    ResponsesUserMessageItemParam,
)
from azure.identity import DefaultAzureCredential

from memory_gym.schemas import MultiSessionOutput

logger = logging.getLogger(__name__)


class FoundryMemoryAgent:
    """Agent using Azure AI Foundry memory store for preference recall.

    This agent:
    1. Creates a memory store (if needed) and populates it with conversation history
    2. Uses the Foundry agent with memory search tool to respond
    3. Memory persists across evaluation tasks for the same session data
    """

    def __init__(
        self,
        memory_store_name: str,
        scope: str = "user",
        chat_model: str = "gpt-4.1-001",
        embedding_model: str = "text-embedding-3-small-001",
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
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self._memory_populated = False
        self._agent = None
        self._conversation_id = None

    def ensure_memory_store(self, multisession_data: MultiSessionOutput, force_recreate: bool = False) -> None:
        """Create memory store if needed and populate with conversation history.

        Args:
            multisession_data: Session data to populate memories from
            force_recreate: If True, delete existing memory store and recreate
        """
        existing_stores = [ms.name for ms in self.client.memory_stores.list()]

        if self.memory_store_name in existing_stores:
            if force_recreate:
                logger.info(f"Deleting existing memory store '{self.memory_store_name}'...")
                self.client.memory_stores.delete(self.memory_store_name)
            else:
                logger.info(f"Memory store '{self.memory_store_name}' already exists")
                self._memory_populated = True
                return

        logger.info(f"Creating memory store '{self.memory_store_name}'...")
        definition = MemoryStoreDefaultDefinition(
            chat_model=self.chat_model,
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
        """Populate memory store with all session conversations."""
        logger.info("Populating memory store with conversation history...")

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
            logger.info(f"Session {session.session_id}: {len(result.memory_operations)} memory operations")

        logger.info("Memory population complete")

    def _ensure_agent(self) -> None:
        """Create or get the Foundry agent with memory search tool."""
        if self._agent is not None:
            return

        tool = MemorySearchTool(
            memory_store_name=self.memory_store_name,
            scope=self.scope,
            update_delay=300,
            search_options=MemorySearchOptions(max_memories=100),
        )

        self._agent = self.client.agents.create_version(
            agent_name="PersonaGymAgent",
            definition=PromptAgentDefinition(
                model=self.chat_model,
                instructions="You are a helpful assistant that remembers user preferences and applies them proactively.",
                tools=[tool],
            ),
        )
        logger.info(f"Created Foundry agent: {self._agent.name}")

    def build_context(self, multisession_data: MultiSessionOutput, force_recreate: bool = False) -> str:
        """Ensure memory store is populated and agent is ready.

        Args:
            multisession_data: Session data to build context from
            force_recreate: If True, delete and recreate memory store
        """
        self.ensure_memory_store(multisession_data, force_recreate=force_recreate)
        self._ensure_agent()
        return "Foundry agent with memory search"

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response using Foundry agent with memory search."""
        if self._agent is None:
            raise ValueError("Agent not initialized. Call build_context first.")

        openai_client = self.client.get_openai_client()

        if self._conversation_id is None:
            conv = openai_client.conversations.create()
            self._conversation_id = conv.id
            logger.debug(f"Created new conversation: {self._conversation_id}")

        last_user_message = ""
        for msg in reversed(conversation):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        response = openai_client.responses.create(
            input=last_user_message,
            conversation=self._conversation_id,
            extra_body={"agent": {"name": self._agent.name, "type": "agent_reference"}},
        )

        return response.output_text

    def reset_conversation(self) -> None:
        """Reset the conversation for a new evaluation task."""
        self._conversation_id = None

    def delete_memory_store(self) -> None:
        """Delete the memory store (for cleanup)."""
        try:
            self.client.memory_stores.delete(self.memory_store_name)
            logger.info(f"Deleted memory store: {self.memory_store_name}")
        except Exception as e:
            logger.warning(f"Failed to delete memory store: {e}")


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


def _search_memories(
    client: AIProjectClient,
    memory_store_name: str,
    scope: str,
    query: str,
    max_memories: int = 5,
) -> list[dict]:
    """Search memories for debugging/inspection."""
    result = client.memory_stores.search_memories(
        name=memory_store_name,
        scope=scope,
        items=[ResponsesUserMessageItemParam(content=query)],
        options=MemorySearchOptions(max_memories=max_memories),
    )

    return [{"id": m.memory_item.memory_id, "content": m.memory_item.content} for m in result.memories]
