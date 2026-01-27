"""
Shared Azure OpenAI client for PersonaGym.

This module provides a simple, reusable Azure OpenAI client for components
that need basic LLM access. Components with more complex needs (like
data_generation's AzureQueryLLM with conversation threading) should keep
their specialized implementations.

Usage:
    from persona_gym.client import get_client, LLMClient

    # Simple query
    client = get_client()
    response = client.complete("What is 2+2?")

    # With custom parameters
    response = client.complete(
        prompt="Tell me a joke",
        max_tokens=100,
        temperature=0.9,
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

import json
import logging
import os
from typing import Any, Optional, TypeVar

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModel)

# Module-level singleton
_client_instance: Optional["LLMClient"] = None


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
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
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

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).

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
            temperature=temperature,
        )

        return response.output_text

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from a list of messages.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).

        Returns:
            The generated text response.
        """
        response = self._client.responses.create(
            model=self.deployment,
            input=messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        return response.output_text

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate a JSON response.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt to set context.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).

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
            temperature=temperature,
            text={"format": {"type": "json_object"}},
        )

        return json.loads(response.output_text)

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
            text_format=response_schema,
            max_output_tokens=max_tokens,
        )

        # The parsed result is guaranteed to match response_schema
        result = response.output_parsed
        assert result is not None, "Structured output parsing returned None"
        return result


def get_client(
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    force_new: bool = False,
) -> LLMClient:
    """Get a shared LLMClient instance (singleton pattern).

    By default, returns the same client instance across calls. This is efficient
    since the Azure OpenAI client handles connection pooling internally.

    Args:
        endpoint: Azure OpenAI endpoint URL.
        deployment: Model deployment name.
        api_version: API version.
        force_new: If True, create a new instance instead of reusing.

    Returns:
        An LLMClient instance.
    """
    global _client_instance

    if force_new or _client_instance is None:
        _client_instance = LLMClient(
            endpoint=endpoint,
            deployment=deployment,
            api_version=api_version,
        )

    return _client_instance


def reset_client() -> None:
    """Reset the singleton client instance.

    Useful for testing or when environment variables change.
    """
    global _client_instance
    _client_instance = None
