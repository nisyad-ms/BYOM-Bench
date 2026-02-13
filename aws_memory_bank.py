"""Standalone AWS Bedrock Memory client.

Provides a simple interface for adding, retrieving, and clearing memories
using AWS Bedrock AgentCore's memory service.

Requires:
    pip install bedrock-agentcore python-dotenv

Environment variables:
    AWS_REGION         — AWS region (default: us-west-2)
    AWS_MEMORY_NAME    — Memory store name (default: StandaloneMemory)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSMemoryBank:
    """Lightweight wrapper around AWS Bedrock AgentCore memories."""

    # Memory extraction is async on the server side; poll until results appear.
    _EXTRACTION_POLL_INTERVAL = 5  # seconds
    _EXTRACTION_MAX_WAIT = 60  # seconds

    def __init__(
        self,
        region_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        event_expiry_days: int = 7,
    ) -> None:
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-west-2")
        self.memory_name = memory_name or os.environ.get("AWS_MEMORY_NAME", "StandaloneMemory")
        self.event_expiry_days = event_expiry_days
        self._memory_id: Optional[str] = None

        from bedrock_agentcore.memory import MemoryClient, MemoryControlPlaneClient

        self.client = MemoryClient(region_name=self.region_name)
        self.control_plane_client = MemoryControlPlaneClient(region_name=self.region_name)
        logger.info("AWSMemoryBank initialized for region: %s", self.region_name)

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def _ensure_memory(self) -> None:
        """Create or reuse the memory store with all three strategies."""
        if self._memory_id is not None:
            return

        from bedrock_agentcore.memory.constants import StrategyType
        from botocore.exceptions import ClientError

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
            memory = self.client.create_memory_and_wait(
                name=self.memory_name,
                description="Standalone long-term memory store",
                strategies=strategies,
                event_expiry_days=self.event_expiry_days,
                max_wait=300,
                poll_interval=10,
            )
            self._memory_id = memory["id"]
            logger.info("Created memory: %s", self._memory_id)

        except ClientError as e:
            if (
                e.response["Error"]["Code"] == "ValidationException"
                and "already exists" in str(e)
            ):
                memories = self.client.list_memories()
                self._memory_id = next(
                    (m["id"] for m in memories if m["name"] == self.memory_name),
                    None,
                )
                if self._memory_id:
                    logger.info("Reusing existing memory: %s", self._memory_id)
                else:
                    raise
            else:
                raise

    @staticmethod
    def _namespaces_for_user(user_id: str) -> List[str]:
        """Return the three namespace paths for a given user."""
        return [
            f"user/{user_id}/preferences",
            f"user/{user_id}/semantic",
            f"user/{user_id}/session/{user_id}/summary",
        ]

    @staticmethod
    def _record_to_dict(record: Dict) -> Dict:
        """Normalise a memory record into a simple dict."""
        result: Dict[str, Any] = {}
        if "content" in record:
            result["content"] = record["content"].get("text", "")
        result["id"] = record.get("memoryRecordId", record.get("id", ""))
        if "namespace" in record:
            result["namespace"] = record["namespace"]
        if "createdAt" in record:
            result["created_at"] = str(record["createdAt"])
        if "updatedAt" in record:
            result["updated_at"] = str(record["updatedAt"])
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memories(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
    ) -> List[Dict]:
        """Store memories extracted from a conversation.

        Args:
            user_id: Unique identifier for the user.
            messages: List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.

        Returns:
            List of memory record dicts (content, id, namespace, timestamps).
        """
        self._ensure_memory()

        # Convert to Bedrock format: list of (content, role) tuples
        formatted: List[tuple[str, str]] = []
        for msg in messages:
            formatted.append((msg["content"], msg["role"]))

        # Snapshot existing record IDs so we can identify new ones
        namespaces = self._namespaces_for_user(user_id)
        existing_ids: set[str] = set()
        last_content = formatted[-1][0]
        for ns in namespaces:
            try:
                records, *_ = self.client.retrieve_memories(
                    memory_id=self._memory_id, namespace=ns, query=last_content, top_k=100,
                )
                for r in records:
                    existing_ids.add(r.get("memoryRecordId", r.get("id", "")))
            except Exception:
                pass

        # Create the event — this triggers async memory extraction
        self.client.create_event(
            memory_id=self._memory_id,
            actor_id=user_id,
            session_id=user_id,
            messages=formatted,
        )

        # Poll for new records to appear
        new_records: List[Dict] = []
        deadline = time.time() + self._EXTRACTION_MAX_WAIT
        while time.time() < deadline:
            time.sleep(self._EXTRACTION_POLL_INTERVAL)
            new_records = []
            for ns in namespaces:
                try:
                    records, *_ = self.client.retrieve_memories(
                        memory_id=self._memory_id, namespace=ns, query=last_content, top_k=100,
                    )
                    for r in records:
                        rid = r.get("memoryRecordId", r.get("id", ""))
                        if rid and rid not in existing_ids:
                            new_records.append(self._record_to_dict(r))
                except Exception:
                    pass
            if new_records:
                break

        logger.info("Stored %d memories for user %s", len(new_records), user_id)
        return new_records

    def retrieve_memories(
        self,
        user_id: str,
        num_memories: int = 10,
        query: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve memories for a user.

        Args:
            user_id: Unique identifier for the user.
            num_memories: Maximum number of memories to return.
            query: If provided, used as the similarity search query.
                   If not provided, a generic query is used.

        Returns:
            List of memory record dicts (content, id, namespace, timestamps).
        """
        self._ensure_memory()

        search_query = query or "user information and preferences"
        namespaces = self._namespaces_for_user(user_id)
        per_ns = max(num_memories // len(namespaces), 1)

        memories: List[Dict] = []
        seen_ids: set[str] = set()

        for ns in namespaces:
            try:
                records, *_ = self.client.retrieve_memories(
                    memory_id=self._memory_id,
                    namespace=ns,
                    query=search_query,
                    top_k=per_ns,
                )
                for r in records:
                    rid = r.get("memoryRecordId", r.get("id", ""))
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        memories.append(self._record_to_dict(r))
            except Exception as e:
                logger.warning("Failed to retrieve from namespace %s: %s", ns, e)

        memories = memories[:num_memories]
        logger.info(
            "Retrieved %d memories for user %s (query=%s)",
            len(memories),
            user_id,
            query,
        )
        return memories

    def clear_memories(self, user_id: str) -> None:  # noqa: ARG002
        """Delete the memory store to clear all memories.

        Note: Bedrock does not currently support per-user memory deletion,
        so this deletes the entire memory store.
        """
        if self._memory_id is None:
            logger.warning("No memory store to delete.")
            return

        logger.warning(
            "Deleting entire memory store (per-user deletion not supported). "
            "All users' memories will be removed."
        )
        self.control_plane_client.delete_memory(memory_id=self._memory_id)
        logger.info("Deleted memory store: %s", self._memory_id)
        self._memory_id = None

    def cleanup(self) -> None:
        """Explicitly clean up the memory store resource."""
        if self._memory_id is None:
            return
        try:
            self.client.delete_memory_and_wait(memory_id=self._memory_id)
            logger.info("Cleaned up memory store: %s", self._memory_id)
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info("Memory store already deleted.")
            else:
                raise
        finally:
            self._memory_id = None


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    bank = AWSMemoryBank()

    sample_messages = [
        {"role": "user", "content": "Hi, my name is Alice and I live in Seattle."},
        {"role": "assistant", "content": "Nice to meet you, Alice! Seattle is a great city."},
        {"role": "user", "content": "I love hiking and prefer dark-roast coffee."},
        {"role": "assistant", "content": "Hiking and dark-roast coffee — great combo!"},
    ]

    user = "alice-demo"

    print("\n--- Adding memories ---")
    records = bank.add_memories(user, sample_messages)
    for r in records:
        print(f"  • {r.get('content', '')}  (namespace={r.get('namespace', '')})")

    print("\n--- Retrieving all memories ---")
    all_mems = bank.retrieve_memories(user, num_memories=10)
    for m in all_mems:
        print(f"  • {m.get('content', '')}  (namespace={m.get('namespace', '')})")

    print("\n--- Similarity search: 'coffee' ---")
    search_mems = bank.retrieve_memories(user, num_memories=5, query="coffee")
    for m in search_mems:
        print(f"  • {m.get('content', '')}  (namespace={m.get('namespace', '')})")

    print("\n--- Cleaning up ---")
    bank.cleanup()
