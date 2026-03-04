"""Foundry Memory Store using Azure AI Foundry memory store (API mode only).

Memory retrieval is done via the Foundry memory store search API.
"""

import itertools
import os
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchOptions,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    ResponsesAssistantMessageItemParam,
    ResponsesSystemMessageItemParam,
    ResponsesUserMessageItemParam,
)
from azure.core.exceptions import HttpResponseError
from azure.core.polling.base_polling import OperationFailed
from azure.identity import DefaultAzureCredential
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory_gym.client import (
    _before_sleep_print,
    _discover_all_endpoints,
    _parse_env_list,
    get_agent_config,
)
from memory_gym.schemas import MultiSessionOutput


def _get_foundry_deployments() -> list[str]:
    """Get Foundry deployment names from AZURE_FOUNDRY_DEPLOYMENTS env var."""
    return _parse_env_list("AZURE_FOUNDRY_DEPLOYMENTS")


def _get_foundry_embedding_deployments() -> list[str]:
    return _parse_env_list("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS")


def get_foundry_configs() -> list[tuple[str, str, str]]:
    """Return all (endpoint, chat_model, embedding_model) triples.

    Discovers endpoints via AZURE_FOUNDRY_ENDPOINT[_N] and pairs each chat
    deployment with an embedding deployment via round-robin.  The result is
    a flat list of triples suitable for round-robin assignment to sessions.

    Raises:
        ValueError: If no Foundry deployments or embeddings are configured.
    """
    chat_pairs = _discover_all_endpoints("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENTS")
    embedding_pairs = _discover_all_endpoints("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS")

    if not chat_pairs:
        primary = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
        if not primary:
            raise ValueError("AZURE_FOUNDRY_ENDPOINT not set in environment")
        chat = _get_foundry_deployments()
        if not chat:
            raise ValueError("AZURE_FOUNDRY_DEPLOYMENTS not set in environment")
        emb = _get_foundry_embedding_deployments()
        if not emb:
            raise ValueError("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS not set in environment")
        return [(primary, c, emb[i % len(emb)]) for i, c in enumerate(chat)]

    # Group embeddings by endpoint for round-robin pairing
    emb_by_endpoint: dict[str, list[str]] = {}
    for ep, dep in embedding_pairs:
        emb_by_endpoint.setdefault(ep, []).append(dep)

    triples: list[tuple[str, str, str]] = []
    for ep, chat_model in chat_pairs:
        emb_list = emb_by_endpoint.get(ep)
        if not emb_list:
            emb_list = _get_foundry_embedding_deployments()
        if not emb_list:
            raise ValueError(
                f"No embedding deployments configured for Foundry endpoint {ep}. "
                "Set AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS environment variable."
            )
        emb_model = emb_list[len(triples) % len(emb_list)]
        triples.append((ep, chat_model, emb_model))

    return triples


_foundry_cfg = get_agent_config("foundry")
_foundry_retry_config = _foundry_cfg["retry"]

# Class-level counter for round-robin chat model selection in create_memory_store
_chat_model_counter = itertools.count()


class FoundryMemoryStore:
    """Memory store backed by Azure AI Foundry memory store API.

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    """

    def __init__(
        self,
        memory_store_name: str,
        scope: str = "user",
        chat_model: str | None = None,
        embedding_model: str | None = None,
        endpoint: str | None = None,
        num_memories: int | None = None,
    ):
        endpoint = endpoint or os.getenv("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_FOUNDRY_ENDPOINT not set in environment")

        self.client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        self.memory_store_name = memory_store_name
        self.scope = scope
        self.num_memories = num_memories if num_memories is not None else _foundry_cfg["num_memories"]

        if chat_model is None:
            deployments = _get_foundry_deployments()
            if not deployments:
                raise ValueError("AZURE_FOUNDRY_DEPLOYMENTS not set in environment")
            chat_model = deployments[next(_chat_model_counter) % len(deployments)]
        self.chat_model = chat_model

        if embedding_model is None:
            emb_deployments = _get_foundry_embedding_deployments()
            if not emb_deployments:
                raise ValueError("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS not set in environment")
            embedding_model = emb_deployments[0]
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Delete any existing memory store, create fresh, and populate with conversation history."""
        existing_stores = [ms.name for ms in self.client.memory_stores.list()]

        if self.memory_store_name in existing_stores:
            self.client.memory_stores.delete(self.memory_store_name)

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

        try:
            self._populate_memories(multisession_data)
        except Exception:
            print(f"Memory population failed — deleting partial store '{self.memory_store_name}'")
            self.client.memory_stores.delete(self.memory_store_name)
            raise

    def retrieve(self, query: str) -> list[str]:
        """Search memories via Foundry memory store API and return fact strings."""
        query_message = ResponsesUserMessageItemParam(content=query)
        search_response = self.client.memory_stores.search_memories(
            name=self.memory_store_name,
            scope=self.scope,
            items=[query_message],  # type: ignore[arg-type]  # SDK accepts this message type at runtime
            options=MemorySearchOptions(max_memories=self.num_memories),
        )
        return [m.memory_item.content for m in search_response.memories]

    def cleanup(self) -> None:
        """Delete the memory store created by this instance."""
        try:
            self.client.memory_stores.delete(self.memory_store_name)
            print(f"Deleted memory store: {self.memory_store_name}")
        except Exception as e:
            if "404" not in str(e) and "NotFound" not in str(e):
                raise
            print(f"Memory store already deleted: {self.memory_store_name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_memories(self, multisession_data: MultiSessionOutput) -> None:
        previous_update_id = None
        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            messages = _to_foundry_messages(session.conversation)
            if not messages:
                continue

            poller = self._update_memories_with_retry(messages, previous_update_id)
            result = poller.result()
            previous_update_id = poller.update_id
            print(f"Session {session.session_id}: {len(result.memory_operations)} memory operations")

    @retry(
        retry=retry_if_exception_type((HttpResponseError, OperationFailed)),
        wait=wait_exponential(
            multiplier=_foundry_retry_config["multiplier"],
            min=_foundry_retry_config["min_seconds"],
            max=_foundry_retry_config["max_seconds"],
        ),
        stop=stop_after_attempt(_foundry_retry_config["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _update_memories_with_retry(self, messages: list, previous_update_id: str | None) -> Any:
        """Call begin_update_memories and wait for result, with retry on transient errors."""
        poller = self.client.memory_stores.begin_update_memories(
            name=self.memory_store_name,
            scope=self.scope,
            items=messages,  # type: ignore[arg-type]  # SDK accepts these message types at runtime
            previous_update_id=previous_update_id,
            update_delay=0,
        )
        poller.result()  # blocks until complete; raises HttpResponseError on server failures
        return poller


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
