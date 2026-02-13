"""
Shared Azure OpenAI client for MemoryGym.

This module provides a simple, reusable Azure OpenAI client for components
that need basic LLM access. Components with more complex needs (like
data_generation's AzureQueryLLM with conversation threading) should keep
their specialized implementations.

Usage:
    from memory_gym.client import LLMClient

    # Chat completion
    client = LLMClient()
    response = client.complete_chat(
        messages=[{"role": "user", "content": "Hello"}],
    )

    # JSON response
    data = client.complete_json(
        prompt="Return a JSON with name and age",
        system_prompt="Return valid JSON only.",
    )
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import yaml
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import APIStatusError, AzureOpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

_config_path = Path(__file__).parent.parent / "configs" / "client_config.yaml"
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_DEFAULT_MAX_TOKENS: int = CONFIG["max_tokens"]["default"]


def _before_sleep_print(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    fn = retry_state.fn
    name = f"{fn.__module__}.{fn.__qualname__}" if fn else "unknown"
    print(f"Retrying {name} in {wait:.1f} seconds as it raised {type(exc).__name__}: {exc}")


_llm_retry = retry(
    stop=stop_after_attempt(CONFIG["retry"]["max_attempts"]),
    wait=wait_exponential(multiplier=1, min=CONFIG["retry"]["wait_seconds"], max=CONFIG["retry"]["wait_seconds"]),
    retry=retry_if_exception_type((APIStatusError, json.JSONDecodeError)),
    before_sleep=_before_sleep_print,
    reraise=True,
)


def _parse_env_list(var_name: str) -> list[str]:
    """Parse a comma-separated environment variable into a list of stripped strings."""
    raw = os.environ.get(var_name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_deployments() -> list[str]:
    """Get list of configured deployments from AZURE_OPENAI_DEPLOYMENTS."""
    return _parse_env_list("AZURE_OPENAI_DEPLOYMENTS")


def _get_default_deployment() -> str:
    """Get default deployment (first from DEPLOYMENTS, or fallback to gpt-4o)."""
    deployments = _get_deployments()
    return deployments[0] if deployments else "gpt-4o"


def _build_input(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Build a message list from a prompt and optional system prompt."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class LeastBusyPool:
    """Mixin providing least-busy selection across a pool of resources.

    Subclasses must set `self._pool_items` (list of resources) before use.
    """

    def __init__(self) -> None:
        self._pool_items: list[Any] = []
        self._in_flight: list[int] = []
        self._pool_lock = threading.Lock()

    def _init_pool(self, items: list[Any]) -> None:
        self._pool_items = items
        self._in_flight = [0] * len(items)

    def _acquire(self) -> tuple[int, Any]:
        with self._pool_lock:
            idx = min(range(len(self._in_flight)), key=lambda i: self._in_flight[i])
            self._in_flight[idx] += 1
            return idx, self._pool_items[idx]

    def _release(self, idx: int) -> None:
        with self._pool_lock:
            self._in_flight[idx] -= 1


class LLMClient:
    """Simple Azure OpenAI client for general-purpose LLM queries.

    This client handles:
    - Azure AD authentication via DefaultAzureCredential
    - Chat completions
    - JSON-formatted responses

    For complex use cases with conversation threading or state management,
    use a specialized client (e.g., AzureQueryLLM in data_generation.py).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
    ):
        """Initialize the Azure OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint URL. Defaults to AZURE_OPENAI_ENDPOINT env var.
            deployment: Model deployment name. Defaults to AZURE_OPENAI_DEPLOYMENT or "gpt-4o".
            api_version: API version. Defaults to AZURE_OPENAI_API_VERSION or "2025-03-01-preview".

        Raises:
            ValueError: If endpoint is not provided and AZURE_OPENAI_ENDPOINT is not set.
        """
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment = deployment or _get_default_deployment()
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT environment variable "
                "or pass endpoint parameter."
            )

        # Set up Azure AD authentication
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

    @_llm_retry
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> str:
        """Generate a completion from a list of messages.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """
        response = self._client.responses.create(
            model=self.deployment,
            input=messages,  # type: ignore[arg-type]  # SDK accepts list[dict] at runtime
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        return response.output_text

    @_llm_retry
    def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        """Generate a JSON response.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in response.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
        """
        response = self._client.responses.create(
            model=self.deployment,
            input=_build_input(prompt, system_prompt),  # type: ignore[arg-type]  # SDK accepts list[dict] at runtime
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}},
        )

        return json.loads(response.output_text)


def _resolve_deployments(deployments: list[str] | None = None) -> list[str]:
    """Resolve deployment list from argument or environment, raising if empty."""
    if deployments is None:
        deployments = _get_deployments()
        if not deployments:
            deployments = [_get_default_deployment()]
    if not deployments:
        raise ValueError("No deployments configured")
    return deployments


class PooledLLMClient(LeastBusyPool):
    """LLM client that routes every call to the least-busy deployment.

    Drop-in replacement for LLMClient that tracks in-flight requests
    per deployment and always picks the one with the fewest active calls.
    This ensures no endpoint sits idle while another is saturated.

    Usage:
        pool = PooledLLMClient()  # Uses AZURE_OPENAI_DEPLOYMENTS env var
        response = pool.complete_chat(messages)  # Routes to least-busy deployment
    """

    def __init__(self, deployments: list[str] | None = None):
        super().__init__()

        deployments = _resolve_deployments(deployments)

        self.deployments = deployments
        self.clients = [LLMClient(deployment=d) for d in deployments]
        self._init_pool(self.clients)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        idx, client = self._acquire()
        try:
            return getattr(client, method)(*args, **kwargs)
        finally:
            self._release(idx)

    def complete_chat(
        self, messages: list[dict[str, str]], max_tokens: int = _DEFAULT_MAX_TOKENS, temperature: float = 1.0
    ) -> str:
        return self._call("complete_chat", messages, max_tokens, temperature)

    def complete_json(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int = _DEFAULT_MAX_TOKENS
    ) -> dict[str, Any]:
        return self._call("complete_json", prompt, system_prompt, max_tokens)


class AsyncLLMPool:
    """Pool of LLM clients across multiple deployments for parallel calls.

    Uses round-robin assignment to distribute work across deployments.
    Each deployment gets its own semaphore to prevent overloading.

    Usage:
        pool = AsyncLLMPool()  # Uses AZURE_OPENAI_DEPLOYMENTS env var

        # Run multiple tasks in parallel
        results = await pool.run_parallel(
            items=[task1, task2, task3],
            func=generate_single_task,  # sync function taking (client, item)
        )
    """

    def __init__(self, deployments: list[str] | None = None):
        """Initialize the pool with multiple deployments.

        Args:
            deployments: List of deployment names. Defaults to AZURE_OPENAI_DEPLOYMENTS env var.
        """
        deployments = _resolve_deployments(deployments)

        self.deployments = deployments
        self._pooled_client = PooledLLMClient(deployments=deployments)

    async def run_parallel(
        self,
        items: list[Any],
        func: Callable[["LLMClient | PooledLLMClient", Any], Any],
        on_result: Callable[[int, Any, Any], None] | None = None,
        max_concurrency: int | None = None,
    ) -> list[Any]:
        """Run a function on multiple items in parallel across deployments.

        Each item's function receives a PooledLLMClient that distributes
        individual LLM calls across all deployments via round-robin.

        Args:
            items: List of items to process.
            func: Sync function taking (client, item) and returning result.
            on_result: Optional callback(index, item, result) called as each completes.
            max_concurrency: Maximum number of concurrent tasks. Defaults to 2x deployments.

        Returns:
            List of results in same order as items.
        """
        if max_concurrency is None:
            max_concurrency = len(self.deployments) * 2

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(items)

        async def process_item(index: int, item: Any) -> None:
            async with semaphore:
                try:
                    result = await asyncio.to_thread(func, self._pooled_client, item)
                    results[index] = result
                    if on_result:
                        on_result(index, item, result)
                except Exception as e:
                    print(f"Task {index} failed: {e}")
                    raise

        tasks = [process_item(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks)
        return results
