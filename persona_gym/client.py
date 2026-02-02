"""
Shared Azure OpenAI client for PersonaGym.

This module provides a simple, reusable Azure OpenAI client for components
that need basic LLM access. Components with more complex needs (like
data_generation's AzureQueryLLM with conversation threading) should keep
their specialized implementations.

Usage:
    from persona_gym.client import LLMClient

    # Simple query
    client = LLMClient()
    response = client.complete("What is 2+2?")

    # With custom parameters
    response = client.complete(
        prompt="Tell me a joke",
        max_tokens=100,
    )

    # With system prompt
    response = client.complete(
        prompt="What's the weather?",
        system_prompt="You are a helpful assistant.",
    )

    # JSON response
    data = client.complete_json(
        prompt="Return a JSON with name and age",
        system_prompt="Return valid JSON only.",
    )
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import APIStatusError, AzureOpenAI
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent / "client_config.yaml"
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_llm_retry = retry(
    stop=stop_after_attempt(CONFIG["retry"]["max_attempts"]),
    wait=wait_exponential(multiplier=1, min=CONFIG["retry"]["wait_seconds"], max=CONFIG["retry"]["wait_seconds"]),
    retry=retry_if_exception_type((APIStatusError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModel)


def _get_deployments() -> list[str]:
    """Get list of configured deployments from AZURE_OPENAI_DEPLOYMENTS."""
    deployments_str = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "")
    if not deployments_str:
        return []
    return [d.strip() for d in deployments_str.split(",") if d.strip()]


def _get_default_deployment() -> str:
    """Get default deployment (first from DEPLOYMENTS, or fallback to gpt-4o)."""
    deployments = _get_deployments()
    return deployments[0] if deployments else "gpt-4o"


class LLMClient:
    """Simple Azure OpenAI client for general-purpose LLM queries.

    This client handles:
    - Azure AD authentication via DefaultAzureCredential
    - Basic chat completions
    - JSON-formatted responses
    - Structured outputs with Pydantic schemas

    For complex use cases with conversation threading or state management,
    use a specialized client (e.g., AzureQueryLLM in data_generation.py).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
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
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

        logger.debug(f"LLMClient initialized: endpoint={self.endpoint}, deployment={self.deployment}")

    @_llm_retry
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """
        input_content = []
        if system_prompt:
            input_content.append({"role": "system", "content": system_prompt})
        input_content.append({"role": "user", "content": prompt})

        response = self._client.responses.create(
            model=self.deployment,
            input=input_content,
            max_output_tokens=max_tokens,
            reasoning={"effort": "high"},
        )

        return response.output_text

    @_llm_retry
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion from a list of messages.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """
        response = self._client.responses.create(
            model=self.deployment,
            input=messages,
            max_output_tokens=max_tokens,
        )

        return response.output_text

    @_llm_retry
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
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
        input_content = []
        if system_prompt:
            input_content.append({"role": "system", "content": system_prompt})
        input_content.append({"role": "user", "content": prompt})

        response = self._client.responses.create(
            model=self.deployment,
            input=input_content,
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}},
        )

        return json.loads(response.output_text)

    @_llm_retry
    def complete_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured response matching a Pydantic schema.

        Uses OpenAI's structured outputs feature to guarantee the response
        conforms to the provided schema.

        Args:
            prompt: The user prompt/question.
            response_schema: A Pydantic BaseModel class defining the expected structure.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in response.

        Returns:
            An instance of response_schema with the parsed data.
        """
        input_content = []
        if system_prompt:
            input_content.append({"role": "system", "content": system_prompt})
        input_content.append({"role": "user", "content": prompt})

        response = self._client.responses.parse(
            model=self.deployment,
            input=input_content,
            response_schema=response_schema,
            max_output_tokens=max_tokens,
        )

        # The parsed result is guaranteed to match response_schema
        result = response.output_parsed
        assert result is not None, "Structured output parsing returned None"
        return result


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

    def __init__(self, deployments: Optional[list[str]] = None):
        """Initialize the pool with multiple deployments.

        Args:
            deployments: List of deployment names. Defaults to AZURE_OPENAI_DEPLOYMENTS env var.
        """
        if deployments is None:
            deployments = _get_deployments()
            if not deployments:
                deployments = [_get_default_deployment()]

        if not deployments:
            raise ValueError("No deployments configured")

        self.deployments = deployments
        self.clients = [LLMClient(deployment=d) for d in deployments]
        self._index = 0
        self._lock = asyncio.Lock()

        logger.info(f"AsyncLLMPool initialized with {len(self.clients)} deployments: {deployments}")

    async def _get_next_client(self) -> LLMClient:
        """Get next client using round-robin."""
        async with self._lock:
            client = self.clients[self._index]
            self._index = (self._index + 1) % len(self.clients)
            return client

    async def run_parallel(
        self,
        items: list[Any],
        func: Callable[[LLMClient, Any], Any],
    ) -> list[Any]:
        """Run a function on multiple items in parallel across deployments.

        Args:
            items: List of items to process.
            func: Sync function taking (client, item) and returning result.

        Returns:
            List of results in same order as items.
        """
        async def process_item(item: Any) -> Any:
            client = await self._get_next_client()
            return await asyncio.to_thread(func, client, item)

        tasks = [process_item(item) for item in items]
        return await asyncio.gather(*tasks)


def get_deployments() -> list[str]:
    """Get list of configured deployments."""
    deployments = _get_deployments()
    return deployments if deployments else [_get_default_deployment()]
