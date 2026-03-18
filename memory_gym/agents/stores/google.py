"""Google Memory Store using Google Vertex AI Agent Engine for memory storage."""

import os
import time
from pathlib import Path
from typing import Any

import vertexai
from google.api_core import exceptions as google_exceptions
from google.genai.errors import ClientError as GenAIClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memory_gym.client import _before_sleep_print, get_agent_config
from memory_gym.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

_google_cfg = get_agent_config("google")


class GoogleMemoryStore(SentinelMixin):
    """Memory store backed by Google Vertex AI Agent Engine.

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    """

    _sentinel_agent_type = "google"

    def __init__(
        self,
        *,
        session_dir: Path,
        user_id: str = "default-user",
        project_id: str | None = None,
        location: str | None = None,
        num_memories: int | None = None,
        sentinel_dir: Path | None = None,
        session_name: str | None = None,
    ):
        self.user_id = user_id
        self.project_id = project_id or os.environ.get("GCLOUD_PROJECT_ID")
        self.location = location or os.environ.get("GCLOUD_LOCATION")
        self.num_memories = num_memories if num_memories is not None else _google_cfg["num_memories"]

        if not self.project_id:
            raise ValueError("project_id must be provided or set via GCLOUD_PROJECT_ID env var")
        if not self.location:
            raise ValueError("location must be provided or set via GCLOUD_LOCATION env var")

        self._vertex_client = vertexai.Client(project=self.project_id, location=self.location)  # type: ignore[attr-defined]  # Client is lazy-loaded via __getattr__
        self._agent_engine: Any = None
        self._agent_engine_name: str | None = None
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Create agent engine and populate memories from conversation history.

        If a valid sentinel exists and the cloud resource is alive, reuses it.
        """
        # Check sentinel for existing valid store
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            store_id = sentinel["store_id"]
            if self._check_store_exists(store_id):
                self._agent_engine_name = store_id
                print(f"Reusing existing Google agent engine: {store_id} (sentinel valid)")
                return
        # Sentinel invalid or store gone — delete stale sentinel
        self._delete_sentinel()

        if self._agent_engine_name is not None:
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
        self._write_sentinel(len(multisession_data.sessions), store_id=self._agent_engine_name)

    def retrieve(self, query: str) -> list[str]:
        """Retrieve fact strings from Google agent engine memory."""
        results = self._retrieve_memories_with_retry(query)
        return [m["fact"] for m in results]

    def cleanup(self) -> None:
        """Delete the agent engine and sentinel to free resources."""
        self._delete_sentinel()
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _google_retry = _google_cfg["retry"]

    _google_retry_decorator = retry(
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

    @_google_retry_decorator
    def _create_engine_with_retry(self, memory_config: dict) -> Any:
        """Create a Vertex AI agent engine with retry logic."""
        return self._vertex_client.agent_engines.create(config={"context_spec": {"memory_bank_config": memory_config}})

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
            if result is not None and result.response is not None and result.response.generated_memories is not None:
                mem_count = len(result.response.generated_memories)

            print(f"Session {session.session_id}: {mem_count} memories generated ({elapsed:.1f}s)")
        total_elapsed = time.time() - total_start
        print(f"Memory population complete: {len(multisession_data.sessions)} sessions in {total_elapsed:.1f}s")

    @_google_retry_decorator
    def _generate_memories_with_retry(self, vertex_messages: list[dict]) -> Any:
        """Call Google memories.generate() with retry logic."""
        return self._vertex_client.agent_engines.memories.generate(
            name=self._agent_engine_name,
            direct_contents_source={"events": vertex_messages},
            scope={"user_id": self.user_id},
            config={"wait_for_completion": True},
        )

    @_google_retry_decorator
    def _retrieve_memories_with_retry(self, query: str) -> list[dict]:
        """Call Google memories.retrieve() with retry logic."""
        top_k = self.num_memories
        results = list(
            self._vertex_client.agent_engines.memories.retrieve(
                name=self._agent_engine_name,
                scope={"user_id": self.user_id},
                similarity_search_params={
                    "search_query": query,
                    "top_k": top_k,
                },
            )
        )
        # Sort by distance ascending (smaller = more similar) as a safety check
        results.sort(key=lambda m: m.distance if m.distance is not None else float("inf"))
        return [{"fact": m.memory.fact} for m in results]

    # ------------------------------------------------------------------
    # Sentinel helpers
    # ------------------------------------------------------------------

    def _check_store_exists(self, store_id: str) -> bool:
        """Check if Google agent engine still exists via lightweight API call."""
        try:
            self._vertex_client.agent_engines.get(name=store_id)
            return True
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                return False
            raise


def _to_vertex_messages(conversation: list[dict[str, str]]) -> list[dict]:
    """Convert conversation to Vertex AI message format."""
    messages = []
    for msg in conversation:
        role = "user" if msg.get("role") in ("user", "system") else "model"
        content = msg.get("content", "")
        messages.append({"content": {"role": role, "parts": [{"text": content}]}})
    return messages
