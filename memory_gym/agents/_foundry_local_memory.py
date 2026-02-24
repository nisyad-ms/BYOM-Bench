"""
Local Foundry Memory implementation for MemoryGym.

Replicates the memory extraction, consolidation, and search logic from
Azure AI Foundry Memory entirely in-process using LanceDB for storage and
Azure OpenAI calls for LLM/embedding operations — no Foundry API needed.

Adapted from foundry-memory/foundry_memory/local_memory.py:
- Sync (not async) to match memory_gym agent convention
- Takes list[dict[str, str]] conversations instead of Thread objects
- Uses Responses API for structured output
- Simplified add/retrieve interface
"""

import datetime
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import lancedb
import lancedb.pydantic
import pandas as pd
import pyarrow as pa
from jinja2 import Template
from openai import AzureOpenAI
from pydantic import BaseModel, BeforeValidator, PlainSerializer, WithJsonSchema
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from memory_gym.client import get_agent_config

_foundry_local_cfg = get_agent_config("foundry_local")

# ---------------------------------------------------------------------------
# Prompt constants — loaded from prompts/foundry_local/
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "foundry_local"

EXTRACT_MEMORIES_SYSTEM_PROMPT = (_PROMPT_DIR / "extraction_system.txt").read_text(encoding="utf-8")
EXTRACT_MEMORIES_USER_PROMPT_TEMPLATE = (_PROMPT_DIR / "extraction_instructions.txt").read_text(encoding="utf-8")
DEFAULT_USER_PROFILE_INSTRUCTIONS_PROMPT = (_PROMPT_DIR / "extraction_user_profile_system.txt").read_text(
    encoding="utf-8"
)
CONSOLIDATE_MEMORY_UPDATES_SYSTEM_PROMPT = (_PROMPT_DIR / "consolidation_system.txt").read_text(encoding="utf-8")
CONSOLIDATE_MEMORY_UPDATES_USER_PROMPT_TEMPLATE = (_PROMPT_DIR / "consolidation_instructions.txt").read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------------------------
# Local type definitions
# ---------------------------------------------------------------------------


class MemoryOperationKind(IntEnum):
    Create = 1
    Update = 2
    Delete = 3


class LocalMemoryItemKind(IntEnum):
    Unspecified = 0
    UserProfile = 1
    ChatSummary = 2


class LocalMemoryKind(IntEnum):
    Profile = 1


def NameEnum(enum: type):  # noqa: N802
    """
    Returns an Annotated type that parses from Enum member names and
    serializes to the Enum member name (not value).
    Port of API-MMI-Engines name_enum.py.
    """
    name_map = dict(enum.__members__)
    valid_names = list(name_map.keys())
    valid_display = ", ".join(valid_names)

    def to_member(v: Any) -> Any:
        if isinstance(v, enum):
            return v
        if isinstance(v, str):
            try:
                return name_map[v]
            except KeyError:
                raise ValueError(f"Invalid {enum.__name__} name {v!r}; valid: {valid_display}") from None
        raise TypeError(f"Expected {enum.__name__} name string, got {type(v).__name__}: {v!r}")

    return Annotated[
        enum,
        BeforeValidator(to_member),
        PlainSerializer(lambda e: e.name, return_type=str, when_used="always"),
        WithJsonSchema({"type": "string", "enum": valid_names}, mode="validation"),
        WithJsonSchema({"type": "string", "enum": valid_names}, mode="serialization"),
    ]


StrLocalMemoryKind = NameEnum(LocalMemoryKind)


@dataclass
class LocalMemoryItem:
    memory_id: str = ""
    updated_at: int = 0
    scope: str = ""
    content: str = ""
    kind: LocalMemoryItemKind = LocalMemoryItemKind.Unspecified
    content_vector: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LanceDB schema
# ---------------------------------------------------------------------------


class LocalMemoryRow(lancedb.pydantic.LanceModel):
    mem_id: str
    scope: str
    src_id: list[str] = []
    kind: int = 0
    time: datetime.datetime
    text: str
    vec: lancedb.pydantic.Vector(_foundry_local_cfg["embedding_dim"], pa.float32(), True) = None  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Internal message type
# ---------------------------------------------------------------------------


@dataclass
class _InternalMessage:
    role: str
    content: list[str]
    time: datetime.datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_strict_json_schema(
    data: bool | dict | float | int | list | str | None,
) -> bool | dict | float | int | list | str | None:
    """Ensure JSON schema is compatible with OpenAI structured outputs."""
    if isinstance(data, dict):
        new_data: dict[str, Any] = {k: i for k, v in data.items() if (i := _ensure_strict_json_schema(v)) is not None}
        # Flatten single-element "allOf" clause
        if (
            isinstance(new_data.get("allOf"), list)
            and len(new_data["allOf"]) == 1
            and isinstance(new_data["allOf"][0], dict)
        ):
            trivial_all_of = new_data["allOf"][0]
            del new_data["allOf"]
            new_data |= trivial_all_of
        if "type" in data and data["type"] == "string":
            if data.get("format") == "date":
                del new_data["format"]
                new_data["description"] = data.get("description", "") + " Return date in YYYY-MM-DD format."
            elif data.get("format") == "time":
                del new_data["format"]
                new_data["description"] = data.get("description", "") + " Return time in HH:MM:SS format."
        if "type" in data and data["type"] == "object":
            new_data["additionalProperties"] = False
            if "properties" in data:
                new_data["required"] = list(data["properties"].keys())
        return new_data
    if isinstance(data, list):
        return [_ensure_strict_json_schema(item) for item in data if item is not None]
    return data


def _esc(val: str) -> str:
    return re.sub(r"[\\\"']", r"\\\g<0>", val)


# ---------------------------------------------------------------------------
# FoundryLocalMemory
# ---------------------------------------------------------------------------


class FoundryLocalMemory:
    """
    In-process memory store replicating the Foundry Memory pipeline:
    extraction -> consolidation -> LanceDB storage, and hybrid search.
    """

    def __init__(
        self,
        completion_client: AzureOpenAI,
        embedding_client: AzureOpenAI,
        completion_model: str,
        embedding_model: str,
        db_path: str = "./.lancedb_foundry_local_memory",
        user_profile_enabled: bool = True,
        chat_summary_enabled: bool = True,
    ):
        self.completion_client = completion_client
        self.embedding_client = embedding_client
        self.completion_model = completion_model
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.user_profile_enabled = user_profile_enabled
        self.chat_summary_enabled = chat_summary_enabled

        # Per-user LanceDB tables
        self._db = lancedb.connect(self.db_path)
        self._user_tables: dict[str, Any] = {}

        # Pre-render user profile prompt
        self._user_profile_prompt = Template(DEFAULT_USER_PROFILE_INSTRUCTIONS_PROMPT).render(
            personal_data_admin="the user or developer",
        )

        # Usage tracking — reset per public call
        self._usage: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, user_id: str, conversation: list[dict[str, str]]) -> int:
        """Full memory pipeline: extract -> consolidate -> apply operations.

        Returns the number of memory operations performed.
        """
        self._usage = {}

        messages = self._conversation_to_internal_messages(conversation)
        if not messages:
            return 0

        scope = user_id
        last_message_time = int(max(m.time.timestamp() for m in messages))

        # 1. Extract memories
        extracted_memories = self._extract_memories(messages, [], last_message_time)

        add_ops: list[dict] = []
        delete_ops: list[dict] = []
        update_ops: list[dict] = []

        # 2. Consolidate user profile memories
        extracted_user_profile = [m for m in extracted_memories if m.kind == LocalMemoryItemKind.UserProfile]
        if extracted_user_profile:
            embeddings = self._embed_texts([m.content for m in extracted_user_profile])
            similar_results: list[list[tuple[LocalMemoryItem, float]]] = []
            for mem, emb in zip(extracted_user_profile, embeddings):
                results = self._search_memories_vector(
                    mem.content,
                    emb,
                    scope,
                    LocalMemoryItemKind.UserProfile,
                    top_k=_foundry_local_cfg["search"]["initial_top_k"],
                )
                similar_results.append(results)

            ranked_similar = self._rank_memories_rrf(similar_results)
            similar_items = [item for item, _ in ranked_similar]

            if similar_items:
                adds, deletes, updates = self._consolidate_memory_operations(
                    extracted_user_profile, similar_items, last_message_time, scope
                )
                add_ops.extend(adds)
                delete_ops.extend(deletes)
                update_ops.extend(updates)
            else:
                # No existing memories — directly add
                for mem in extracted_user_profile:
                    add_ops.append(
                        {
                            "operation": MemoryOperationKind.Create,
                            "kind": mem.kind.name,
                            "content": mem.content,
                            "_memory_item": LocalMemoryItem(
                                memory_id=uuid.uuid4().hex,
                                updated_at=last_message_time,
                                scope=scope,
                                content=mem.content,
                                kind=mem.kind,
                            ),
                        }
                    )

        # 3. Directly add chat summary memories
        for mem in extracted_memories:
            if mem.kind == LocalMemoryItemKind.ChatSummary:
                add_ops.append(
                    {
                        "operation": MemoryOperationKind.Create,
                        "kind": mem.kind.name,
                        "content": mem.content,
                        "_memory_item": LocalMemoryItem(
                            memory_id=uuid.uuid4().hex,
                            updated_at=last_message_time,
                            scope=scope,
                            content=mem.content,
                            kind=mem.kind,
                        ),
                    }
                )

        # 4. Apply operations to LanceDB
        self._apply_memory_operations(add_ops, delete_ops, update_ops, scope, last_message_time)

        return len(add_ops) + len(update_ops) + len(delete_ops)

    def retrieve(self, user_id: str, query: str, top_k: int | None = None) -> list[str]:
        """Contextual retrieval: hybrid search using query string."""
        self._usage = {}
        scope = user_id
        resolved_top_k: int = top_k if top_k is not None else _foundry_local_cfg["search"]["default_top_k"]

        if not query or not query.strip():
            return []

        # Embed query
        query_embeddings = self._embed_texts([query])
        query_vector = query_embeddings[0] if query_embeddings else None

        # Hybrid search
        results = self._search_memories_vector(query, query_vector, scope, kind_filter=None, top_k=resolved_top_k)
        return [item.content for item, _ in results]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _conversation_to_internal_messages(self, conversation: list[dict[str, str]]) -> list[_InternalMessage]:
        """Convert memory_gym conversation format to internal messages."""
        msgs = []
        msg_time = datetime.datetime.now(datetime.timezone.utc)
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue
            msgs.append(_InternalMessage(role=role, content=[content], time=msg_time))
        return msgs

    def _extract_memories(
        self,
        messages: list[_InternalMessage],
        context: list[_InternalMessage],
        last_message_time: int,
    ) -> list[LocalMemoryItem]:
        """Extract memory items from messages using LLM structured output."""
        if not messages:
            return []

        # Dynamic Pydantic schema
        TurnSchema: type[IntEnum] = IntEnum("TurnSchema", {f"msg_{i}": i for i in range(len(messages))})  # type: ignore[assignment]

        class Segment(BaseModel):
            """A memory segment."""

            detailed_summary: str
            user_information: list[str]
            turn_ids: list[TurnSchema]  # type: ignore[valid-type]

        class MemorySegments(BaseModel):
            segments: list[Segment]

        # Render prompts
        system_prompt = Template(EXTRACT_MEMORIES_SYSTEM_PROMPT).render(
            user_profile_instructions=self._user_profile_prompt,
        )
        user_prompt = Template(EXTRACT_MEMORIES_USER_PROMPT_TEMPLATE).render(
            context_messages=context,
            messages=messages,
            datetime=datetime.datetime,
            timezone=datetime.timezone,
        )

        # LLM call
        segments_response = self._llm_parse(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            MemorySegments,
        )

        # Convert to LocalMemoryItems
        user_profile_items: list[LocalMemoryItem] = (
            []
            if not self.user_profile_enabled
            else [
                LocalMemoryItem(kind=LocalMemoryItemKind.UserProfile, updated_at=last_message_time, content=memory)
                for segment in segments_response.segments
                for content in segment.user_information
                if (memory := content.strip())
            ]
        )
        chat_summary_items: list[LocalMemoryItem] = (
            []
            if not self.chat_summary_enabled
            else [
                LocalMemoryItem(kind=LocalMemoryItemKind.ChatSummary, updated_at=last_message_time, content=memory)
                for segment in segments_response.segments
                if (memory := segment.detailed_summary.strip())
            ]
        )
        return user_profile_items + chat_summary_items

    def _consolidate_memory_operations(
        self,
        extracted_memories: list[LocalMemoryItem],
        similar_memories: list[LocalMemoryItem],
        last_message_time: int,
        scope: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Consolidate extracted and similar memories into add/update/delete operations."""

        # Dynamic Pydantic schemas
        ExistingMemoryId: type[IntEnum] = IntEnum("memory_id", {f"mem_{i}": i for i in range(len(similar_memories))})  # type: ignore[assignment]
        NewMemoryId: type[IntEnum] = IntEnum(  # type: ignore[assignment]
            "new_memory_id",
            {f"mem_{i}": i for i in range(len(similar_memories), len(similar_memories) + len(extracted_memories))},
        )

        class OperationName(StrEnum):
            Add = "add"
            Update = "update"
            Delete = "delete"

        class AddMemory(BaseModel):
            """Add a new memory."""

            name: Literal[OperationName.Add]
            memory_kind: StrLocalMemoryKind  # type: ignore[valid-type]
            memory: str

        class UpdateMemory(BaseModel):
            """Update an existing memory."""

            name: Literal[OperationName.Update]
            memory_id: ExistingMemoryId  # type: ignore[valid-type]
            memory: str

        class DeleteMemory(BaseModel):
            """Delete an existing memory."""

            name: Literal[OperationName.Delete]
            memory_id: ExistingMemoryId  # type: ignore[valid-type]

        class MemoryOperation(BaseModel):
            """A single operation. Provide a short few-word reason for the operation and the IDs of each new memory that motivated the operation."""

            reason: str
            operation: AddMemory | UpdateMemory | DeleteMemory
            source_memory_ids: list[NewMemoryId]  # type: ignore[valid-type]

        class MemoryOperationList(BaseModel):
            """Operations to add, update, or delete memories."""

            operations: list[MemoryOperation]

        # Render prompt
        user_prompt = Template(CONSOLIDATE_MEMORY_UPDATES_USER_PROMPT_TEMPLATE).render(
            extracted_memories=extracted_memories,
            similar_memories=similar_memories,
            datetime=datetime.datetime,
            timezone=datetime.timezone,
        )

        consolidate_operations: list[MemoryOperation] = (
            self._llm_parse(
                [
                    {"role": "system", "content": CONSOLIDATE_MEMORY_UPDATES_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                MemoryOperationList,
            )
        ).operations

        # Deduplicate operations
        unique_adds: dict[str, AddMemory] = {
            op.memory: op
            for gnd_op in consolidate_operations
            if isinstance(op := gnd_op.operation, AddMemory) and op.memory
        }
        unique_deletes: dict[int, DeleteMemory] = {
            op.memory_id.value: op
            for gnd_op in consolidate_operations
            if isinstance(op := gnd_op.operation, DeleteMemory)
        }
        unique_updates: dict[int, UpdateMemory] = {
            op.memory_id.value: op
            for gnd_op in consolidate_operations
            if isinstance(op := gnd_op.operation, UpdateMemory)
        }

        # Dedup: empty update -> deletion, skip no-op, prefer deletion over update
        unique_deletes |= {
            k: DeleteMemory(name=OperationName.Delete, memory_id=k) for (k, v) in unique_updates.items() if not v.memory
        }
        unique_updates = {
            k: v
            for (k, v) in unique_updates.items()
            if v.memory and v.memory != similar_memories[k].content and k not in unique_deletes
        }

        # Convert to operation dicts
        adds = [
            {
                "operation": MemoryOperationKind.Create,
                "kind": LocalMemoryItemKind(op.memory_kind).name,
                "content": memory,
                "_memory_item": LocalMemoryItem(
                    memory_id=uuid.uuid4().hex,
                    updated_at=last_message_time,
                    scope=scope,
                    content=memory,
                    kind=LocalMemoryItemKind(op.memory_kind),
                ),
            }
            for (memory, op) in unique_adds.items()
        ]
        deletes = [
            {
                "operation": MemoryOperationKind.Delete,
                "kind": similar_memories[memory_idx].kind.name,
                "content": similar_memories[memory_idx].content,
                "_memory_item": similar_memories[memory_idx],
            }
            for memory_idx in unique_deletes
        ]
        updates = [
            {
                "operation": MemoryOperationKind.Update,
                "kind": similar_memories[memory_idx].kind.name,
                "content": op.memory,
                "_memory_item": LocalMemoryItem(
                    memory_id=similar_memories[memory_idx].memory_id,
                    updated_at=last_message_time,
                    scope=similar_memories[memory_idx].scope,
                    content=op.memory,
                    kind=similar_memories[memory_idx].kind,
                ),
            }
            for (memory_idx, op) in unique_updates.items()
        ]

        return adds, deletes, updates

    _retry_cfg = _foundry_local_cfg["retry"]

    @retry(
        stop=stop_after_attempt(_retry_cfg["max_attempts"]),
        wait=wait_fixed(_retry_cfg["wait_fixed"]),
        retry=retry_if_exception_type(Exception),
    )
    def _llm_parse(self, messages: list[dict], schema: type) -> Any:
        """Call LLM with structured output and parse response."""
        strict_schema = _ensure_strict_json_schema(schema.model_json_schema())

        response = self.completion_client.responses.create(
            model=self.completion_model,
            input=messages,  # type: ignore[arg-type]
            text={  # type: ignore[arg-type]
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": strict_schema,
                    "strict": True,
                }
            },
            temperature=_foundry_local_cfg["extraction"]["temperature"],
        )

        resp_str = response.output_text
        if not resp_str:
            raise ValueError("LLM returned empty content")
        return schema.model_validate_json(resp_str)

    @retry(
        stop=stop_after_attempt(_retry_cfg["max_attempts"]),
        wait=wait_fixed(_retry_cfg["wait_fixed"]),
        retry=retry_if_exception_type(Exception),
    )
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the embedding client."""
        if not texts:
            return []
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=texts)
        return [data.embedding for data in response.data]

    def _get_or_create_user_table(self, scope: str) -> Any:
        """Get or create a LanceDB table for the user scope."""
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", scope)
        if table_name in self._user_tables:
            return self._user_tables[table_name]

        try:
            table = self._db.open_table(table_name)
        except Exception:
            table = self._db.create_table(table_name, schema=LocalMemoryRow)  # type: ignore[arg-type]
        self._user_tables[table_name] = table
        return table

    def _search_memories_vector(
        self,
        query: str | None,
        vector: list[float] | None,
        scope: str,
        kind_filter: LocalMemoryItemKind | None,
        top_k: int = 5,
    ) -> list[tuple[LocalMemoryItem, float]]:
        """Hybrid search combining vector + FTS + RRF reranker."""
        table = self._get_or_create_user_table(scope)

        try:
            row_count = table.count_rows()
        except Exception:
            return []
        if row_count == 0:
            return []

        expr = table.search(query_type="hybrid")

        # Build where clause
        where_parts = [f"scope = '{_esc(scope)}'"]
        if kind_filter is not None:
            where_parts.append(f"kind = {kind_filter.value}")
        where_clause = " AND ".join(where_parts)
        expr = expr.where(where_clause)

        if query:
            expr = expr.text(query)
        if vector:
            expr = expr.vector(vector)

        expr = expr.limit(top_k)

        try:
            df: pd.DataFrame = expr.to_pandas()
        except Exception:
            # Fall back to filter-only query
            try:
                df = table.search().where(where_clause).limit(top_k).to_pandas()
            except Exception:
                return []

        results = []
        for row in df.itertuples():  # type: ignore[reportAttributeAccessIssue]
            item = LocalMemoryItem(
                memory_id=row.mem_id,  # type: ignore[union-attr]
                updated_at=int(row.time.timestamp()) if hasattr(row.time, "timestamp") else 0,  # type: ignore[union-attr]
                scope=row.scope,  # type: ignore[union-attr]
                content=row.text,  # type: ignore[union-attr]
                kind=LocalMemoryItemKind(row.kind),  # type: ignore[union-attr]
                content_vector=row.vec.tolist() if hasattr(row, "vec") and row.vec is not None and len(row.vec) else [],  # type: ignore[union-attr]
            )
            score = getattr(row, "_relevance_score", getattr(row, "_distance", getattr(row, "_score", 0.0)))
            results.append((item, float(score) if score is not None else 0.0))
        return results

    def _rank_memories_rrf(
        self,
        search_results: list[list[tuple[LocalMemoryItem, float]]],
    ) -> list[tuple[LocalMemoryItem, float]]:
        """RRF reranking with k=60, deduplication by memory_id."""
        memory_items: dict[str, LocalMemoryItem] = {}
        memory_scores: dict[str, float] = {}

        for results in search_results:
            for rank, (item, _) in enumerate(results):
                mid = item.memory_id
                memory_items[mid] = item
                memory_scores[mid] = memory_scores.get(mid, 0.0) + 1.0 / (rank + 60)

        sorted_memories = sorted(
            [(memory_items[mid], score) for mid, score in memory_scores.items()],
            key=lambda x: (x[1], x[0].updated_at),
            reverse=True,
        )
        return sorted_memories

    def _apply_memory_operations(
        self,
        adds: list[dict],
        deletes: list[dict],
        updates: list[dict],
        scope: str,
        updated_at: int,
    ) -> None:
        """Apply memory operations to LanceDB."""
        all_ops = adds + updates
        if not all_ops and not deletes:
            return

        table = self._get_or_create_user_table(scope)

        # Embed content for adds and updates
        items_to_embed = [op["_memory_item"] for op in all_ops if op.get("_memory_item")]
        if items_to_embed:
            contents = [item.content for item in items_to_embed]
            embeddings = self._embed_texts(contents)
            for item, emb in zip(items_to_embed, embeddings):
                item.content_vector = emb
                item.updated_at = updated_at

        # Apply adds
        add_items = [op["_memory_item"] for op in adds if op.get("_memory_item")]
        if add_items:
            rows = [
                {
                    "mem_id": item.memory_id,
                    "scope": scope,
                    "src_id": [],
                    "kind": item.kind.value,
                    "time": datetime.datetime.fromtimestamp(item.updated_at, datetime.timezone.utc),
                    "text": item.content,
                    "vec": item.content_vector if item.content_vector else None,
                }
                for item in add_items
            ]
            table.add(rows)

        # Apply deletes
        delete_items = [op["_memory_item"] for op in deletes if op.get("_memory_item")]
        if delete_items:
            mem_ids = [item.memory_id for item in delete_items]
            id_list = ",".join([f"'{_esc(mid)}'" for mid in mem_ids])
            table.delete(f"scope = '{_esc(scope)}' AND mem_id IN ({id_list})")

        # Apply updates
        update_items = [op["_memory_item"] for op in updates if op.get("_memory_item")]
        for item in update_items:
            update_values: dict[str, Any] = {
                "text": item.content,
                "time": datetime.datetime.fromtimestamp(item.updated_at, datetime.timezone.utc),
            }
            if item.content_vector:
                update_values["vec"] = item.content_vector
            table.update(where=f"mem_id = '{_esc(item.memory_id)}'", values=update_values)

        # Rebuild FTS index
        try:
            table.create_fts_index("text", replace=True)
        except Exception:
            pass

    def cleanup(self) -> None:
        """Remove the LanceDB database directory."""
        try:
            if self.db_path and os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"Cleaned up LanceDB at {self.db_path}")
        except Exception as e:
            print(f"Failed to cleanup LanceDB: {e}")
