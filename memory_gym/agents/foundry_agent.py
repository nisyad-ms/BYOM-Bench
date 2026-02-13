"""Foundry Memory Agent using Azure AI Foundry memory store."""

import json
import os
import threading
import uuid
from typing import Any

import openai
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentVersionDetails,
    MemorySearchOptions,
    MemorySearchTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
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

from memory_gym.client import CONFIG, LeastBusyPool, PooledLLMClient, _before_sleep_print, _parse_env_list
from memory_gym.prompts import render_prompt
from memory_gym.schemas import MultiSessionOutput

_SEARCH_MEMORIES_TOOL = {
    "type": "function",
    "name": "search_user_memories",
    "description": "Search stored memories about the user's preferences, habits, and personal information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories about the user.",
            },
        },
        "required": ["query"],
    },
}


def _get_foundry_deployments() -> list[str]:
    """Get Foundry deployment names from AZURE_FOUNDRY_DEPLOYMENTS env var."""
    return _parse_env_list("AZURE_FOUNDRY_DEPLOYMENTS")


def _get_foundry_embedding_deployments() -> list[str]:
    return _parse_env_list("AZURE_FOUNDRY_EMBEDDINGS_DEPLOYMENTS")


def get_foundry_embedding_models() -> list[str]:
    return _get_foundry_embedding_deployments() or ["text-embedding-3-small-001"]


_foundry_retry_config = CONFIG["retry_foundry"]


class FoundryMemoryAgent(LeastBusyPool):
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
        super().__init__()

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
            embedding_model = get_foundry_embedding_models()[0]
        self.embedding_model = embedding_model
        self._memory_populated = False
        self._init_lock = threading.Lock()

        if chat_models is None:
            chat_models = _get_foundry_deployments() or ["gpt-4.1-001"]
        self.chat_models = chat_models
        self._agents: list[AgentVersionDetails] = []
        self._agent_ids = [str(uuid.uuid4())[:8] for _ in chat_models]
        self._local = threading.local()

    def create_memory_store(self, multisession_data: MultiSessionOutput) -> None:
        """Delete any existing memory store, create fresh, and populate with conversation history.

        Thread-safe: only one thread will create/populate the store.
        """
        with self._init_lock:
            if self._memory_populated:
                return

            existing_stores = [ms.name for ms in self.client.memory_stores.list()]

            if self.memory_store_name in existing_stores:
                self.client.memory_stores.delete(self.memory_store_name)

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

            try:
                self._populate_memories(multisession_data)
            except Exception:
                print(f"Memory population failed — deleting partial store '{self.memory_store_name}'")
                self.client.memory_stores.delete(self.memory_store_name)
                raise
            self._memory_populated = True

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
                        instructions=render_prompt("agents/agent_system_foundry_tool"),
                        tools=[tool],
                    ),
                )
                self._agents.append(agent)

            self._init_pool(self._agents)

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Ensure memory store is populated and agents are ready."""
        self.create_memory_store(multisession_data)
        self._ensure_agents()
        return "Foundry agent with memory search"

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response using least-busy Foundry agent.

        Thread-safe: each thread gets its own conversation ID per agent index.
        """
        if not self._agents:
            raise ValueError("Agents not initialized. Call build_context first.")

        idx, agent = self._acquire()
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
            self._release(idx)

    _foundry_retry = CONFIG["retry_foundry"]

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
        wait=wait_exponential(
            multiplier=_foundry_retry["multiplier"],
            min=_foundry_retry["min_seconds"],
            max=_foundry_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_foundry_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _call_with_retry(
        self, openai_client: openai.OpenAI, message: str, conversation_id: str, agent: AgentVersionDetails
    ) -> Any:
        """Call responses.create with retry logic for rate limits."""
        return openai_client.responses.create(
            input=message,
            conversation=conversation_id,
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
        )

    def cleanup(self) -> None:
        """Delete all agents and the memory store created by this instance."""
        for agent in self._agents:
            try:
                self.client.agents.delete(agent.name)
                print(f"Deleted agent: {agent.name}")
            except Exception as e:
                if "404" not in str(e) and "NotFound" not in str(e):
                    raise
                print(f"Agent already deleted: {agent.name}")
        self._agents.clear()

        try:
            self.client.memory_stores.delete(self.memory_store_name)
            print(f"Deleted memory store: {self.memory_store_name}")
        except Exception as e:
            if "404" not in str(e) and "NotFound" not in str(e):
                raise
            print(f"Memory store already deleted: {self.memory_store_name}")
        self._memory_populated = False

    def reset_conversation(self) -> None:
        """Reset all conversation IDs for the current thread."""
        for i in range(len(self.chat_models)):
            setattr(self._local, f"conversation_id_{i}", None)


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


class FoundryMemoryAPIAgent(FoundryMemoryAgent):
    """Agent using Azure AI Foundry memory store via direct API calls.

    Instead of attaching MemorySearchTool to a Foundry prompt agent,
    this agent manually searches memories via the memory store API and
    uses a tool-calling loop (like GoogleMemoryAgent) so we control
    memory retrieval.
    """

    def __init__(
        self,
        memory_store_name: str,
        scope: str = "user",
        chat_models: list[str] | None = None,
        embedding_model: str | None = None,
        num_memories: int = 10,
    ):
        super().__init__(
            memory_store_name=memory_store_name,
            scope=scope,
            chat_models=chat_models,
            embedding_model=embedding_model,
        )
        self.num_memories = num_memories
        self._llm_client: PooledLLMClient | None = None

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Ensure memory store is populated and LLM client pool is ready."""
        self.create_memory_store(multisession_data)
        if self._llm_client is None:
            self._llm_client = PooledLLMClient()
        return "Foundry agent with memory API"

    def _search_memories(self, query: str) -> list[dict[str, str]]:
        """Search memories via Foundry memory store API."""
        query_message = ResponsesUserMessageItemParam(content=query)
        search_response = self.client.memory_stores.search_memories(
            name=self.memory_store_name,
            scope=self.scope,
            items=[query_message],  # type: ignore[arg-type]  # SDK accepts this message type at runtime
            options=MemorySearchOptions(max_memories=self.num_memories),
        )
        return [{"fact": m.memory_item.content} for m in search_response.memories]

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response using Azure OpenAI with Foundry memory search via API.

        Uses tool-calling: the LLM decides when to search memories.
        """
        if self._llm_client is None:
            raise ValueError("LLM client not initialized. Call build_context first.")

        system_prompt = render_prompt("agents/agent_system_foundry_api")

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        idx, llm_client = self._llm_client._acquire()
        try:
            azure_client = llm_client._client
            return self._respond_with_tools(azure_client, llm_client.deployment, messages)
        finally:
            self._llm_client._release(idx)

    _azure_retry = CONFIG["retry"]

    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, json.JSONDecodeError)
        ),
        wait=wait_exponential(
            multiplier=1,
            min=_azure_retry["wait_seconds"],
            max=_azure_retry["wait_seconds"],
        ),
        stop=stop_after_attempt(_azure_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _respond_with_tools(
        self, azure_client: openai.AzureOpenAI, deployment: str, messages: list[dict[str, Any]]
    ) -> str:
        """Make Azure OpenAI call with tool-calling loop."""
        response = azure_client.responses.create(
            model=deployment,
            input=messages,  # type: ignore[arg-type]
            tools=[_SEARCH_MEMORIES_TOOL],  # type: ignore[list-item]
            max_output_tokens=CONFIG["max_tokens"]["agent"],
        )

        while True:
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                break

            tool_results: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                if tool_call.name == "search_user_memories":
                    args = json.loads(tool_call.arguments)
                    query = args["query"]
                    memories = self._search_memories(query)
                    output = json.dumps(memories, ensure_ascii=False)
                else:
                    output = json.dumps({"error": f"Unknown tool: {tool_call.name}"})

                tool_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": output,
                    }
                )

            response = azure_client.responses.create(
                model=deployment,
                input=response.output + tool_results,  # type: ignore[arg-type,operator]
                tools=[_SEARCH_MEMORIES_TOOL],  # type: ignore[list-item]
                max_output_tokens=CONFIG["max_tokens"]["agent"],
            )

        return response.output_text

    def reset_conversation(self) -> None:
        """No-op: no per-conversation state to reset."""
