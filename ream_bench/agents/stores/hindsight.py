"""Hindsight Memory Store for REAM-Bench.

Uses the Hindsight library (https://github.com/vectorize-io/hindsight) with an
embedded PostgreSQL database for biomimetic memory organization (world facts,
experiences, observations) and multi-strategy retrieval (semantic + keyword +
graph + temporal with reciprocal rank fusion).

Azure OpenAI provides LLM and embedding operations.

Install: pip install ream-bench[hindsight]
"""

import os
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from ream_bench.client import get_agent_config, resolve_azure_openai_config
from ream_bench.schemas import MultiSessionOutput

from ._sentinel import SentinelMixin

_hindsight_cfg = get_agent_config("hindsight")

# Azure AD scope for OpenAI
_AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


def _get_azure_token() -> str:
    """Get a bearer token for Azure OpenAI.

    Uses AZURE_TOKEN env var if set (e.g. in Docker), otherwise DefaultAzureCredential.
    """
    env_token = os.environ.get("AZURE_TOKEN")
    if env_token:
        return env_token
    credential = DefaultAzureCredential()
    token = credential.get_token(_AZURE_OPENAI_SCOPE)
    return token.token


# ---------------------------------------------------------------------------
# Monkey-patches for Azure OpenAI compatibility.
#
# Hindsight creates standard OpenAI / AsyncOpenAI clients which don't add the
# api-version query param Azure requires. We patch both:
#   1. OpenAIEmbeddings.initialize — replace with AzureOpenAI client
#   2. AsyncOpenAI.__init__ — inject default_query={"api-version": ...}
# ---------------------------------------------------------------------------

_patched = False
_azure_api_version: str = ""
_azure_emb_endpoint: str = ""
_azure_emb_deployment: str = ""


def _apply_azure_patches(api_version: str, emb_endpoint: str, emb_deployment: str) -> None:
    global _patched, _azure_api_version, _azure_emb_endpoint, _azure_emb_deployment  # noqa: PLW0603
    if _patched:
        return
    _azure_api_version = api_version
    _azure_emb_endpoint = emb_endpoint
    _azure_emb_deployment = emb_deployment

    # --- Patch 1: OpenAIEmbeddings → use AzureOpenAI client ---
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    async def _azure_emb_initialize(self: Any) -> None:
        if self._client is not None:
            return
        from openai import AzureOpenAI

        env_token = os.environ.get("AZURE_TOKEN")
        if env_token:
            self._client = AzureOpenAI(
                azure_endpoint=_azure_emb_endpoint,
                api_key=env_token,
                api_version=_azure_api_version,
            )
        else:
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, _AZURE_OPENAI_SCOPE)
            self._client = AzureOpenAI(
                azure_endpoint=_azure_emb_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=_azure_api_version,
            )
        response = self._client.embeddings.create(model=_azure_emb_deployment, input=["test"])
        if response.data:
            self._dimension = len(response.data[0].embedding)

    OpenAIEmbeddings.initialize = _azure_emb_initialize  # type: ignore[assignment]

    # --- Patch 2: AsyncOpenAI → inject default_query for api-version ---
    from openai import AsyncOpenAI

    _original_init = AsyncOpenAI.__init__

    def _patched_init(self: Any, **kwargs: Any) -> None:
        if "default_query" not in kwargs:
            kwargs["default_query"] = {"api-version": _azure_api_version}
        _original_init(self, **kwargs)

    AsyncOpenAI.__init__ = _patched_init  # type: ignore[assignment]

    _patched = True


class HindsightMemoryStore(SentinelMixin):
    """Memory store backed by Hindsight (embedded PostgreSQL + multi-strategy retrieval).

    Implements the ``MemoryStore`` protocol: ``populate``, ``retrieve``, ``cleanup``.
    Uses Azure OpenAI for LLM/embedding and Hindsight's embedded PostgreSQL for storage.
    """

    _sentinel_agent_type = "hindsight"

    def __init__(
        self,
        *,
        session_dir: Path,
        bank_id: str = "default-user",
        num_memories: int | None = None,
        sentinel_dir: Path | None = None,
        session_name: str | None = None,
    ):
        from hindsight import HindsightClient, HindsightServer

        # Discover Azure OpenAI endpoints
        endpoint, chat_deployment, emb_endpoint, emb_deployment, api_version = resolve_azure_openai_config()

        # Get Azure AD bearer token
        azure_token = _get_azure_token()

        # Apply Azure compatibility patches before server starts
        _apply_azure_patches(api_version, emb_endpoint, emb_deployment)

        # Configure embedding env vars — Hindsight reads these to select provider
        os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai"
        os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"] = emb_deployment
        os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] = azure_token

        # Disable local BERT cross-encoder reranker — use pure RRF scoring instead.
        # The default "local" provider downloads ms-marco-MiniLM-L-6-v2 (~80MB) which
        # adds significant startup latency in cold environments (Docker, CI).
        os.environ["HINDSIGHT_API_RERANKER_PROVIDER"] = "rrf"

        # Start embedded server (bundles PostgreSQL)
        # db_url="pg0" uses embedded PostgreSQL — no external DB needed
        self._server = HindsightServer(
            db_url="pg0",
            llm_provider="openai",
            llm_model=chat_deployment,
            llm_api_key=azure_token,
            llm_base_url=f"{endpoint}/openai/deployments/{chat_deployment}",
        )
        self._server.start(timeout=120)
        self._client = HindsightClient(base_url=self._server.url)

        self.bank_id = bank_id
        self.profile = session_dir.name
        self.num_memories = num_memories if num_memories is not None else _hindsight_cfg["num_memories"]
        self._sentinel_dir = sentinel_dir
        self._session_name = session_name

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    def populate(self, multisession_data: MultiSessionOutput) -> None:
        """Ingest all sessions into Hindsight via its retain() extraction pipeline.

        If a valid sentinel exists, skips re-ingestion (the embedded PG persists
        across server restarts within the same profile).
        """
        sentinel = self._read_sentinel()
        if sentinel and sentinel["sessions_ingested"] == len(multisession_data.sessions):
            print(f"Reusing existing Hindsight bank '{self.bank_id}' (sentinel valid)")
            return
        self._delete_sentinel()

        # Delete stale bank before populating
        try:
            self._client.banks.delete(bank_id=self.bank_id)
        except Exception:
            pass  # Bank may not exist yet

        for session in multisession_data.sessions:
            if not session.conversation:
                continue

            # Format conversation as "role: content" lines
            lines = []
            for msg in session.conversation:
                lines.append(f"{msg['role']}: {msg['content']}")
            content = "\n".join(lines)

            self._client.retain(
                bank_id=self.bank_id,
                content=content,
                context=f"session-{session.session_id}",
                document_id=f"session-{session.session_id}",
            )
            print(f"Session {session.session_id}: retained into Hindsight", flush=True)

        self._write_sentinel(len(multisession_data.sessions), bank_id=self.bank_id)

    def retrieve(self, query: str) -> list[str]:
        """Search Hindsight memories and return fact strings."""
        response = self._client.recall(
            bank_id=self.bank_id,
            query=query,
            max_tokens=4096,
        )
        return [r.text for r in response.results[: self.num_memories]]

    def cleanup(self) -> None:
        """Delete the Hindsight bank and shut down the embedded server."""
        self._delete_sentinel()
        try:
            self._client.banks.delete(bank_id=self.bank_id)
        except Exception:
            pass  # Best-effort cleanup
        try:
            self._server.stop()
        except Exception:
            pass
