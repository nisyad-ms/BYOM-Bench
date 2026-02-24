"""AWS Memory Agent using AWS Bedrock AgentCore for memory storage."""

import os
import re
import threading
import time
from typing import Any

from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from memory_gym.client import PooledLLMClient, _before_sleep_print, get_agent_config
from memory_gym.schemas import MultiSessionOutput

from ._tool_calling import respond_with_memory_search

_aws_cfg = get_agent_config("aws")

# Bedrock memory extraction is async; poll until results appear.
_EXTRACTION_POLL_INTERVAL: int = _aws_cfg["polling"]["extraction_interval"]
_EXTRACTION_MAX_WAIT: int = _aws_cfg["timeouts"]["extraction_max_wait"]

# Non-retryable error codes from AWS — permanent failures that shouldn't be retried.
_NON_RETRYABLE_CODES = {"AccessDeniedException", "UnauthorizedAccess", "ValidationException"}


def _is_retryable_client_error(exc: BaseException) -> bool:
    """Return True if the ClientError is transient and worth retrying."""
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code", "")
    return code not in _NON_RETRYABLE_CODES


class AWSMemoryAgent:
    """Agent using AWS Bedrock AgentCore for memory storage and Azure OpenAI for LLM responses.

    Creates a Bedrock memory store to store/retrieve memories, while using
    Azure OpenAI for chat responses with a search_memories tool.
    """

    def __init__(
        self,
        memory_name: str,
        user_id: str = "default-user",
        region_name: str | None = None,
        num_memories: int | None = None,
        event_expiry_days: int | None = None,
    ):
        self.user_id = user_id
        self.region_name = region_name or os.environ.get("AWS_REGION")
        if not self.region_name:
            raise ValueError("AWS_REGION not set in environment")
        # AWS memory names must match [a-zA-Z][a-zA-Z0-9_]{0,47}
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", memory_name)
        if not sanitized[0].isalpha():
            sanitized = f"mg_{sanitized}"
        self.memory_name = sanitized[:48]
        self.num_memories = num_memories if num_memories is not None else _aws_cfg["num_memories"]
        self.event_expiry_days = event_expiry_days if event_expiry_days is not None else _aws_cfg["event_expiry_days"]

        from bedrock_agentcore.memory import MemoryClient, MemoryControlPlaneClient

        self._memory_client = MemoryClient(region_name=self.region_name)
        self._control_plane_client = MemoryControlPlaneClient(region_name=self.region_name)
        self._llm_client = PooledLLMClient()
        self._memory_id: str | None = None
        self._memory_populated = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def _create_memory_store(self) -> None:
        """Create a fresh memory store with all three strategies.

        If a store with the same name already exists (orphan from a prior
        failed run), deletes it first and creates fresh.
        Sets self._memory_id as early as possible so cleanup can always find
        the store even if the wait-for-active step fails.
        """
        if self._memory_id is not None:
            return

        from bedrock_agentcore.memory.constants import StrategyType

        strategies = [
            {
                StrategyType.USER_PREFERENCE.value: {
                    "name": "UserPreferences",
                    "description": "Captures user preferences and personal information",
                    "namespaces": ["user/{actorId}/preferences"],
                }
            },
            {
                StrategyType.SEMANTIC.value: {
                    "name": "SemanticMemory",
                    "description": "Stores semantic understanding of conversations",
                    "namespaces": ["user/{actorId}/semantic"],
                }
            },
            {
                StrategyType.SUMMARY.value: {
                    "name": "ConversationSummary",
                    "description": "Maintains conversation summaries and context",
                    "namespaces": ["user/{actorId}/session/{sessionId}/summary"],
                }
            },
        ]

        try:
            self._create_and_wait(strategies)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException" and "already exists" in str(e):
                # Delete the existing store (orphan from a prior failed run) and create fresh.
                memories = self._memory_client.list_memories()
                existing_id = next(
                    (m["id"] for m in memories if m["id"].startswith(self.memory_name)),
                    None,
                )
                if existing_id:
                    print(f"Deleting existing AWS memory store before fresh create: {existing_id}")
                    self._memory_id = existing_id
                    self._cleanup_memory_store()
                    self._create_and_wait(strategies)
                else:
                    raise
            else:
                raise

    _aws_retry = _aws_cfg["retry"]

    @retry(
        retry=retry_if_exception(_is_retryable_client_error),
        wait=wait_exponential(
            multiplier=_aws_retry["multiplier"],
            min=_aws_retry["min_seconds"],
            max=_aws_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_aws_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _create_and_wait(self, strategies: list[dict]) -> None:
        """Create memory store, capture its ID immediately, then wait for ACTIVE.

        Sets self._memory_id right after the create call so cleanup can always
        find the store, even if the wait-for-active step fails or times out.
        """
        memory = self._memory_client.create_memory(
            name=self.memory_name,
            description="MemoryGym long-term memory store",
            strategies=strategies,
            event_expiry_days=self.event_expiry_days,
        )
        memory_id = memory.get("memoryId", memory.get("id", ""))
        self._memory_id = memory_id
        print(f"Created AWS memory store: {memory_id}, waiting for ACTIVE...")

        # Poll until ACTIVE (or FAILED / timeout)
        max_wait = _aws_cfg["timeouts"]["memory_store_creation"]
        poll_interval = _aws_cfg["polling"]["creation_interval"]
        start = time.time()
        while time.time() - start < max_wait:
            status = self._memory_client.get_memory_status(memory_id)
            if status == "ACTIVE":
                print(f"AWS memory store {memory_id} is ACTIVE")
                return
            if status == "FAILED":
                raise RuntimeError(f"AWS memory store creation failed: {memory_id}")
            time.sleep(poll_interval)

        raise TimeoutError(f"AWS memory store {memory_id} did not become ACTIVE within {max_wait}s")

    @staticmethod
    def _namespaces_for_user(user_id: str) -> list[str]:
        """Return the three namespace paths for a given user."""
        return [
            f"user/{user_id}/preferences",
            f"user/{user_id}/semantic",
            f"user/{user_id}/session/{user_id}/summary",
        ]

    # ------------------------------------------------------------------
    # Memory population
    # ------------------------------------------------------------------

    def _populate_memories(self, multisession_data: MultiSessionOutput) -> None:
        """Generate memories from all sessions by sending conversation events."""
        total_start = time.time()
        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            messages = _to_bedrock_messages(session.conversation)
            if not messages:
                continue

            t0 = time.time()
            new_records = self._add_memories_for_session(messages)
            elapsed = time.time() - t0
            print(f"Session {session.session_id}: {len(new_records)} memories extracted ({elapsed:.1f}s)")

        total_elapsed = time.time() - total_start
        print(f"Memory population complete: {len(multisession_data.sessions)} sessions in {total_elapsed:.1f}s")

    @retry(
        retry=retry_if_exception(_is_retryable_client_error),
        wait=wait_exponential(
            multiplier=_aws_retry["multiplier"],
            min=_aws_retry["min_seconds"],
            max=_aws_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_aws_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _add_memories_for_session(self, messages: list[tuple[str, str]]) -> list[dict]:
        """Send conversation event and poll for extracted memories."""
        assert self._memory_id is not None
        namespaces = self._namespaces_for_user(self.user_id)
        last_content = messages[-1][0]

        # Snapshot existing record IDs so we can identify new ones
        existing_ids: set[str] = set()
        internal_top_k = _aws_cfg["search"]["internal_top_k"]
        for ns in namespaces:
            try:
                records = self._memory_client.retrieve_memories(
                    memory_id=self._memory_id,
                    namespace=ns,
                    query=last_content,
                    top_k=internal_top_k,
                )
                for r in records:
                    existing_ids.add(_get_record_id(r))
            except Exception as e:
                print(f"Warning: failed to snapshot existing records in {ns}: {e}")

        # Create the event - triggers async memory extraction
        self._memory_client.create_event(
            memory_id=self._memory_id,
            actor_id=self.user_id,
            session_id=self.user_id,
            messages=messages,
        )

        # Poll for new records to appear
        new_records: list[dict] = []
        deadline = time.time() + _EXTRACTION_MAX_WAIT
        while time.time() < deadline:
            time.sleep(_EXTRACTION_POLL_INTERVAL)
            new_records = []
            for ns in namespaces:
                try:
                    records = self._memory_client.retrieve_memories(
                        memory_id=self._memory_id,
                        namespace=ns,
                        query=last_content,
                        top_k=internal_top_k,
                    )
                    for r in records:
                        rid = _get_record_id(r)
                        if rid and rid not in existing_ids:
                            new_records.append(_record_to_dict(r))
                except Exception as e:
                    print(f"Warning: failed to poll records in {ns}: {e}")
            if new_records:
                break

        return new_records

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable_client_error),
        wait=wait_exponential(
            multiplier=_aws_retry["multiplier"],
            min=_aws_retry["min_seconds"],
            max=_aws_retry["max_seconds"],
        ),
        stop=stop_after_attempt(_aws_retry["max_attempts"]),
        before_sleep=_before_sleep_print,
    )
    def _retrieve_memories(self, query: str) -> list[dict[str, str]]:
        """Retrieve memories across all namespaces for the user."""
        assert self._memory_id is not None
        namespaces = self._namespaces_for_user(self.user_id)
        per_ns = max(self.num_memories // len(namespaces), 1)

        memories: list[dict[str, str]] = []
        seen_ids: set[str] = set()

        for ns in namespaces:
            try:
                records = self._memory_client.retrieve_memories(
                    memory_id=self._memory_id,
                    namespace=ns,
                    query=query,
                    top_k=per_ns,
                )
                for r in records:
                    rid = _get_record_id(r)
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        content = _get_record_content(r)
                        if content:
                            memories.append({"fact": content})
            except Exception as e:
                print(f"Warning: failed to retrieve memories from {ns}: {e}")

        return memories[: self.num_memories]

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def build_context(self, multisession_data: MultiSessionOutput) -> str:
        """Create memory store and populate memories from conversation history.

        Thread-safe: only one thread will create/populate the store.
        Cleans up on populate failure as defense-in-depth; the caller
        (test_evaluation.py) also owns cleanup via try/finally.
        """
        with self._init_lock:
            if self._memory_populated:
                return "AWS agent with memory search"

            self._create_memory_store()

            try:
                self._populate_memories(multisession_data)
            except Exception:
                print(f"Memory population failed — cleaning up AWS memory store '{self.memory_name}'")
                self._cleanup_memory_store()
                raise

            self._memory_populated = True

        return "AWS agent with memory search"

    def respond(self, conversation: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Generate a response using Azure OpenAI with AWS memory search tool."""
        if self._memory_id is None:
            raise ValueError("Memory store not initialized. Call build_context first.")
        return respond_with_memory_search(
            self._llm_client, "agents/agent_system_memory", conversation, self._retrieve_memories
        )

    def reset_conversation(self) -> None:
        """No-op: no per-conversation state to reset."""

    def cleanup(self) -> None:
        """Delete the memory store to free resources."""
        self._cleanup_memory_store()

    _CLEANUP_POLL_INTERVAL: int = _aws_cfg["polling"]["cleanup_interval"]
    _CLEANUP_MAX_WAIT: int = _aws_cfg["timeouts"]["cleanup_max_wait"]

    def _cleanup_memory_store(self) -> None:
        """Delete the AWS memory store, retrying if it's in a transitional state."""
        if self._memory_id is None:
            return
        memory_id = self._memory_id
        self._memory_id = None
        self._memory_populated = False

        deadline = time.time() + self._CLEANUP_MAX_WAIT
        while True:
            try:
                self._memory_client.delete_memory_and_wait(memory_id=memory_id)
                print(f"Cleaned up AWS memory store: {memory_id}")
                return
            except Exception as e:
                msg = str(e).lower()
                if "not found" in msg or "404" in str(e):
                    print("AWS memory store already deleted.")
                    return
                if "transitional state" in msg and time.time() < deadline:
                    print(f"AWS memory store in transitional state, retrying delete: {memory_id}")
                    time.sleep(self._CLEANUP_POLL_INTERVAL)
                    continue
                raise


def _to_bedrock_messages(conversation: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Convert conversation to Bedrock format: list of (content, role) tuples.

    Bedrock AgentCore expects uppercase roles: USER, ASSISTANT.
    """
    messages: list[tuple[str, str]] = []
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("system", "user"):
            role = "USER"
        else:
            role = "ASSISTANT"
        messages.append((content, role))
    return messages


def _get_record_id(record: Any) -> str:
    """Extract the record ID from a Bedrock memory record."""
    if isinstance(record, dict):
        return record.get("memoryRecordId", record.get("id", ""))
    return getattr(record, "memoryRecordId", getattr(record, "id", ""))


def _get_record_content(record: Any) -> str:
    """Extract text content from a Bedrock memory record."""
    if isinstance(record, dict):
        content = record.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)
    content = getattr(record, "content", None)
    if content is not None and hasattr(content, "text"):
        return content.text
    return str(content) if content else ""


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Normalise a memory record into a simple dict."""
    result: dict[str, Any] = {}
    result["content"] = _get_record_content(record)
    result["id"] = _get_record_id(record)
    if isinstance(record, dict):
        if "namespace" in record:
            result["namespace"] = record["namespace"]
        if "createdAt" in record:
            result["created_at"] = str(record["createdAt"])
        if "updatedAt" in record:
            result["updated_at"] = str(record["updatedAt"])
    else:
        if hasattr(record, "namespace"):
            result["namespace"] = record.namespace
        if hasattr(record, "createdAt"):
            result["created_at"] = str(record.createdAt)
        if hasattr(record, "updatedAt"):
            result["updated_at"] = str(record.updatedAt)
    return result
