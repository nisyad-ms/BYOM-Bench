"""Google Memory Agent using Google Vertex AI Agent Engine for memory storage."""

import json
import os
import threading
import time
from typing import Any

import openai
import vertexai
from google.api_core import exceptions as google_exceptions
from google.genai.errors import ClientError as GenAIClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory_gym.client import CONFIG, PooledLLMClient, _before_sleep_print
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


class GoogleMemoryAgent:
    """Agent using Google Vertex AI Agent Engine for memory storage and Azure OpenAI for LLM responses.

    Creates a Vertex AI agent engine to store/retrieve memories, while using
    Azure OpenAI for chat responses with a search_memories tool.
    """

    def __init__(
        self,
        user_id: str = "default-user",
        project_id: str | None = None,
        location: str | None = None,
        num_memories: int = 10,
    ):
        self.user_id = user_id
        self.project_id = project_id or os.environ.get("GCLOUD_PROJECT_ID")
        self.location = location or os.environ.get("GCLOUD_LOCATION", "us-central1")
        self.num_memories = num_memories

        if not self.project_id:
            raise ValueError("project_id must be provided or set via GCLOUD_PROJECT_ID env var")

        self._vertex_client = vertexai.Client(project=self.project_id, location=self.location)  # type: ignore[attr-defined]  # Client is lazy-loaded via __getattr__
        self._llm_client = PooledLLMClient()
        self._agent_engine: Any = None
        self._agent_engine_name: str | None = None
        self._memory_populated = False
        self._init_lock = threading.Lock()

    def build_context(self, multisession_data: MultiSessionOutput, force_recreate: bool = False) -> str:
        """Create agent engine and populate memories from conversation history.

        Thread-safe: only one thread will create/populate the engine.
        """
        with self._init_lock:
            if self._memory_populated and not force_recreate:
                return "Google agent with memory search"

            if self._agent_engine_name is not None and force_recreate:
                try:
                    self._vertex_client.agent_engines.delete(name=self._agent_engine_name, force=True)
                except Exception as e:
                    if "404" not in str(e) and "NOT_FOUND" not in str(e):
                        raise
                self._agent_engine = None
                self._agent_engine_name = None

            if self._agent_engine is None:
                memory_config: dict = {
                    "customization_configs": [
                        {
                            "memory_topics": [
                                {"managed_memory_topic": {"managed_topic_enum": "USER_PERSONAL_INFO"}},
                                {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
                                {"managed_memory_topic": {"managed_topic_enum": "KEY_CONVERSATION_DETAILS"}},
                                {"managed_memory_topic": {"managed_topic_enum": "EXPLICIT_INSTRUCTIONS"}},
                            ]
                        }
                    ]
                }
                self._agent_engine = self._create_engine_with_retry(memory_config)
                self._agent_engine_name = self._agent_engine.api_resource.name
                print(f"Created Google agent engine: {self._agent_engine_name}")

            self._populate_memories(multisession_data)
            self._memory_populated = True

        return "Google agent with memory search"

    _google_retry = CONFIG["retry_google"]

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
                GenAIClientError,
            )
        ),
        wait=wait_exponential(
            multiplier=_google_retry["multiplier"],
            min=_google_retry["min_seconds"],
            max=_google_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_google_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _create_engine_with_retry(self, memory_config: dict) -> Any:
        """Create a Vertex AI agent engine with retry logic."""
        return self._vertex_client.agent_engines.create(
            config={"context_spec": {"memory_bank_config": memory_config}}
        )

    def _populate_memories(self, multisession_data: MultiSessionOutput) -> None:
        """Generate memories from all sessions."""
        total_start = time.time()
        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            vertex_messages = _to_vertex_messages(session.conversation)
            if not vertex_messages:
                continue

            t0 = time.time()
            result = self._generate_memories_with_retry(vertex_messages)
            elapsed = time.time() - t0

            mem_count = 0
            if result is not None and result.response is not None:
                mem_count = len(result.response.generated_memories)

            print(f"Session {session.session_id}: {mem_count} memories generated ({elapsed:.1f}s)")
        total_elapsed = time.time() - total_start
        print(f"Memory population complete: {len(multisession_data.sessions)} sessions in {total_elapsed:.1f}s")

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
                GenAIClientError,
            )
        ),
        wait=wait_exponential(
            multiplier=_google_retry["multiplier"],
            min=_google_retry["min_seconds"],
            max=_google_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_google_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _generate_memories_with_retry(self, vertex_messages: list[dict]) -> Any:
        """Call Google memories.generate() with retry logic."""
        return self._vertex_client.agent_engines.memories.generate(
            name=self._agent_engine_name,
            direct_contents_source={"events": vertex_messages},
            scope={"user_id": self.user_id},
            config={"wait_for_completion": True},
        )

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
                GenAIClientError,
            )
        ),
        wait=wait_exponential(
            multiplier=_google_retry["multiplier"],
            min=_google_retry["min_seconds"],
            max=_google_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_google_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _retrieve_memories_with_retry(self, query: str) -> list[dict]:
        """Call Google memories.retrieve() with retry logic."""
        top_k = self.num_memories
        results = self._vertex_client.agent_engines.memories.retrieve(
            name=self._agent_engine_name,
            scope={"user_id": self.user_id},
            similarity_search_params={
                "search_query": query,
                "top_k": top_k,
            },
        )
        return [{"fact": m.memory.fact} for m in results]

    def respond(self, conversation: list[dict[str, str]]) -> str:
        """Generate a response using Azure OpenAI with Google memory search tool.

        Uses tool-calling: the LLM decides when to search memories.
        """
        if self._agent_engine_name is None:
            raise ValueError("Agent engine not initialized. Call build_context first.")

        system_prompt = render_prompt("agents/agent_system_google")

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Get the underlying AzureOpenAI client from the least-busy pool member
        idx, llm_client = self._llm_client._acquire()
        try:
            azure_client = llm_client._client
            return self._respond_with_tools(azure_client, llm_client.deployment, messages)
        finally:
            self._llm_client._release(idx)

    _azure_retry = CONFIG["retry"]

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
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

        # Tool-calling loop: handle function_call items
        while True:
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                break

            # Build follow-up input with tool results
            tool_results: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                if tool_call.name == "search_user_memories":
                    args = json.loads(tool_call.arguments)
                    query = args["query"]
                    memories = self._retrieve_memories_with_retry(query)
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

    def cleanup(self) -> None:
        """Delete the agent engine to free resources."""
        if self._agent_engine_name is None:
            return
        try:
            self._vertex_client.agent_engines.delete(name=self._agent_engine_name, force=True)
            print(f"Cleaned up Google agent engine: {self._agent_engine_name}")
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                print("Google agent engine already deleted.")
            else:
                raise
        finally:
            self._agent_engine = None
            self._agent_engine_name = None


def _to_vertex_messages(conversation: list[dict[str, str]]) -> list[dict]:
    """Convert conversation to Vertex AI message format."""
    messages = []
    for msg in conversation:
        role = "user" if msg.get("role") in ("user", "system") else "model"
        content = msg.get("content", "")
        messages.append({"content": {"role": role, "parts": [{"text": content}]}})
    return messages
