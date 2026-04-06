"""Graphiti (Zep) Memory Store for REAM-Bench.

Uses the graphiti-core library (https://github.com/getzep/graphiti) with an
embedded Kùzu graph database for temporal knowledge graph construction and
retrieval. Azure OpenAI provides LLM and embedding operations.
"""

import asyncio
import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from openai import AsyncAzureOpenAI

from ream_bench.client import get_agent_config, resolve_azure_openai_config
from ream_bench.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

_zep_cfg = get_agent_config("zep")

# ---------------------------------------------------------------------------
# Monkey-patch: graphiti-core 0.28.1 AzureOpenAILLMClient._handle_structured_response
# returns dict instead of tuple[dict, int, int], causing ValueError in generate_response.
# Patch it to return the expected 3-tuple with token counts.
# ---------------------------------------------------------------------------
_original_handle_structured = AzureOpenAILLMClient._handle_structured_response


def _patched_handle_structured_response(self: AzureOpenAILLMClient, response: Any) -> Any:
    result = _original_handle_structured(self, response)
    if isinstance(result, dict):
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        return result, input_tokens, output_tokens
    return result


AzureOpenAILLMClient._handle_structured_response = _patched_handle_structured_response  # type: ignore[assignment]


class ZepMemoryStore(SentinelMixin):
    """Memory store backed by Graphiti temporal knowledge graph + Kùzu.

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    Uses Azure OpenAI for LLM/embedding and local Kùzu for graph storage.
    """

    _sentinel_agent_type = "zep"

    def __init__(
        self,
        *,
        session_dir: Path,
        num_memories: int | None = None,
        sentinel_dir: Path | None = None,
        session_name: str | None = None,
    ):
        # Discover Azure OpenAI endpoints
        endpoint, chat_deployment, _emb_endpoint, emb_deployment, api_version = resolve_azure_openai_config()

        # Private event loop — Graphiti is fully async but MemoryStore protocol is sync.
        # The evaluation runner calls store methods via asyncio.to_thread() from the main
        # event loop, so a dedicated loop avoids conflicts.
        self._loop = asyncio.new_event_loop()

        # Build async Azure OpenAI client with DefaultAzureCredential
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

        llm_client = AzureOpenAILLMClient(
            azure_client=azure_client,
            config=LLMConfig(model=chat_deployment, small_model=chat_deployment),
        )
        embedder = AzureOpenAIEmbedderClient(azure_client=azure_client, model=emb_deployment)
        cross_encoder = OpenAIRerankerClient(config=LLMConfig(model=chat_deployment), client=azure_client)

        db_path = f".kuzu/{session_dir.name}"

        # Ensure parent directory exists for Kùzu DB
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        driver = KuzuDriver(db=db_path)
        # Workaround: graphiti-core accesses driver._database in add_episode but
        # KuzuDriver doesn't set it (Neo4j-only attribute). Set it to avoid AttributeError.
        driver._database = session_name or session_dir.name  # type: ignore[attr-defined]
        self._graphiti = Graphiti(
            graph_driver=driver, llm_client=llm_client, embedder=embedder, cross_encoder=cross_encoder
        )

        self.db_path = db_path
        self.num_memories = num_memories if num_memories is not None else _zep_cfg["num_memories"]
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name
        self._group_id = session_name or session_dir.name
        self._azure_client = azure_client
        self._chat_deployment = chat_deployment

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Ingest all sessions into the Graphiti knowledge graph.

        If a valid sentinel exists and the Kùzu store is on disk, reuses it.
        """
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            if os.path.exists(self.db_path):
                print(f"Reusing existing Graphiti store at {self.db_path} (sentinel valid)")
                return
        self._delete_sentinel()

        # Delete stale store before populating
        if os.path.exists(self.db_path):
            self._remove_db_files()
            print(f"Removed stale Graphiti store at {self.db_path}")
            # Re-create driver + graphiti after deleting DB
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            driver = KuzuDriver(db=self.db_path)
            driver._database = self._group_id  # type: ignore[attr-defined]
            self._graphiti = Graphiti(
                graph_driver=driver,
                llm_client=self._graphiti.llm_client,
                embedder=self._graphiti.embedder,
                cross_encoder=OpenAIRerankerClient(
                    config=LLMConfig(model=self._chat_deployment), client=self._azure_client
                ),
            )

        self._loop.run_until_complete(self._graphiti.build_indices_and_constraints())
        self._create_kuzu_fts_indices()

        for i, session in enumerate(multisession_data.sessions):
            if not session.conversation:
                continue

            lines = []
            for msg in session.conversation:
                lines.append(f"{msg['role']}: {msg['content']}")
            episode_body = "\n".join(lines)

            self._loop.run_until_complete(
                self._graphiti.add_episode(
                    name=f"session_{i}",
                    episode_body=episode_body,
                    source=EpisodeType.text,
                    source_description="multi-session conversation",
                    reference_time=datetime.now(timezone.utc),
                    group_id=self._group_id,
                )
            )
            print(f"Session {session.session_id}: ingested into Graphiti (episode {i})", flush=True)

        self._write_sentinel(len(multisession_data.sessions), store_path=self.db_path)

    def retrieve(self, query: str) -> list[str]:
        """Search the Graphiti knowledge graph and return fact strings."""
        results: list[EntityEdge] = self._loop.run_until_complete(
            self._graphiti.search(query, group_ids=[self._group_id], num_results=self.num_memories)
        )
        return [edge.fact for edge in results]

    def cleanup(self) -> None:
        """Close Graphiti client and remove local Kùzu store."""
        self._delete_sentinel()
        try:
            self._loop.run_until_complete(self._graphiti.close())
        except Exception:
            pass  # Best-effort cleanup
        if os.path.exists(self.db_path):
            self._remove_db_files()
        self._loop.close()

    # ------------------------------------------------------------------
    # Kùzu DB file helpers
    # ------------------------------------------------------------------

    def _create_kuzu_fts_indices(self) -> None:
        """Create FTS indices that KuzuDriver.build_indices_and_constraints omits."""
        import kuzu

        conn = kuzu.Connection(self._graphiti.driver.db)  # type: ignore[attr-defined]
        for query in get_fulltext_indices(GraphProvider.KUZU):
            try:
                conn.execute(query)
            except RuntimeError:
                pass  # Index already exists
        conn.close()

    def _remove_db_files(self) -> None:
        """Remove Kùzu database files (db_path, .wal, .lock)."""
        for path in glob.glob(f"{self.db_path}*"):
            os.remove(path)
