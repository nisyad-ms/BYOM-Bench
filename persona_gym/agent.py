#!/usr/bin/env python3
"""
agent.py - Agent implementations for TOD evaluation

This module defines agents that can be evaluated on task-oriented dialogues:
- BaseAgent: Abstract base class defining the agent interface
- ContextAwareAgent: Has access to user's conversation history (upper bound)
- NoContextAgent: No access to past conversations (lower bound)

Future additions:
- MemoryAgent: Agent with its own memory harness for retrieval
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for agents that can be evaluated on TOD tasks.

    All agents must implement the `respond` method which takes:
    - conversation_history: The current dialogue turns
    - user_message: The latest user message
    - available_tools: Tools the agent can use
    - context_messages: Optional past conversation history (may be ignored)

    Returns a dict with:
    - "content": str - The text response
    - "tool_calls": List[Dict] - Optional tool calls [{"name": str, "params": dict}]
    """

    @abstractmethod
    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        available_tools: Dict[str, Dict],
        context_messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response given conversation history and user message."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name for logging/identification."""
        ...

    @property
    @abstractmethod
    def uses_context(self) -> bool:
        """Return whether this agent uses past conversation context."""
        ...


# =============================================================================
# Azure OpenAI Base Mixin
# =============================================================================

class AzureOpenAIMixin:
    """Mixin providing Azure OpenAI client setup."""

    def _init_azure_client(
        self,
        deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Initialize Azure OpenAI client with DefaultAzureCredential."""
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
        )
        self.deployment = deployment

    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """Parse tool calls from response content."""
        tool_calls = []
        try:
            if "tool_call" in content.lower():
                json_match = re.search(r'\{[\s\S]*"tool_call"[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "tool_call" in parsed:
                        tool_calls.append(parsed["tool_call"])
        except (json.JSONDecodeError, Exception):
            pass
        return tool_calls

    def _call_llm(self, messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Make a call to Azure OpenAI and return the content."""
        response = self.client.responses.create(
            model=self.deployment,
            input=messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return response.output_text


# =============================================================================
# Context-Aware Agent (Upper Bound)
# =============================================================================

class ContextAwareAgent(BaseAgent, AzureOpenAIMixin):
    """
    Agent with full access to user's past conversation history.

    This represents the UPPER BOUND of performance - the agent has access
    to all past conversations and can use that context to personalize
    responses and anticipate user preferences.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_context_turns: int = 50,
    ):
        """
        Initialize the context-aware agent.

        Args:
            deployment: Azure OpenAI deployment name
            endpoint: Azure OpenAI endpoint
            system_prompt: Custom system prompt (uses default if None)
            max_context_turns: Maximum number of past conversation turns to include
        """
        self._init_azure_client(deployment, endpoint)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_turns = max_context_turns

    def _default_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to the user's conversation history. 
You should use information from past conversations to personalize your responses and complete tasks efficiently.
When the user asks you to do something, proactively use their known preferences without requiring reminders.
If you need to use a tool, respond with a JSON object containing "tool_call" with "name" and "params"."""

    @property
    def name(self) -> str:
        return "ContextAwareAgent"

    @property
    def uses_context(self) -> bool:
        return True

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        available_tools: Dict[str, Dict],
        context_messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response using past conversation context."""

        messages = [{"role": "system", "content": self.system_prompt}]

        # Add context from past conversations (this is what makes it "context-aware")
        if context_messages:
            context_str = "\n".join([
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in context_messages[-self.max_context_turns:]
            ])
            messages.append({
                "role": "system",
                "content": f"Here is the user's conversation history for context:\n\n{context_str}"
            })
            logger.debug(f"ContextAwareAgent: Using {len(context_messages[-self.max_context_turns:])} context turns")

        # Add available tools info
        if available_tools:
            tools_str = json.dumps(available_tools, indent=2)
            messages.append({
                "role": "system",
                "content": f"You have access to these tools:\n{tools_str}\n\nTo use a tool, include in your response: {{\"tool_call\": {{\"name\": \"tool_name\", \"params\": {{...}}}}}}"
            })

        # Add conversation history
        for turn in conversation_history:
            messages.append(turn)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call the model
        content = self._call_llm(messages)
        tool_calls = self._parse_tool_calls(content)

        return {
            "content": content,
            "tool_calls": tool_calls,
        }


# =============================================================================
# No-Context Agent (Lower Bound)
# =============================================================================

class NoContextAgent(BaseAgent, AzureOpenAIMixin):
    """
    Agent WITHOUT access to user's past conversation history.

    This represents the LOWER BOUND of performance - the agent has no
    memory of past interactions and must rely solely on the current
    conversation to complete tasks. It cannot anticipate preferences.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the no-context agent.

        Args:
            deployment: Azure OpenAI deployment name
            endpoint: Azure OpenAI endpoint
            system_prompt: Custom system prompt (uses default if None)
        """
        self._init_azure_client(deployment, endpoint)
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are a helpful AI assistant. 
Complete the user's requests to the best of your ability.
If you need to use a tool, respond with a JSON object containing "tool_call" with "name" and "params"."""

    @property
    def name(self) -> str:
        return "NoContextAgent"

    @property
    def uses_context(self) -> bool:
        return False

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        available_tools: Dict[str, Dict],
        context_messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response WITHOUT using past conversation context."""

        messages = [{"role": "system", "content": self.system_prompt}]

        # NOTE: Deliberately ignoring context_messages - this is the key difference
        if context_messages:
            logger.debug(f"NoContextAgent: Ignoring {len(context_messages)} context messages")

        # Add available tools info
        if available_tools:
            tools_str = json.dumps(available_tools, indent=2)
            messages.append({
                "role": "system",
                "content": f"You have access to these tools:\n{tools_str}\n\nTo use a tool, include in your response: {{\"tool_call\": {{\"name\": \"tool_name\", \"params\": {{...}}}}}}"
            })

        # Add conversation history (only current dialogue, not past conversations)
        for turn in conversation_history:
            messages.append(turn)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call the model
        content = self._call_llm(messages)
        tool_calls = self._parse_tool_calls(content)

        return {
            "content": content,
            "tool_calls": tool_calls,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_agent(
    agent_type: str,
    deployment: Optional[str] = None,
    endpoint: Optional[str] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create agents by type.

    Args:
        agent_type: One of "context", "no_context" (future: "memory")
        deployment: Azure OpenAI deployment name
        endpoint: Azure OpenAI endpoint
        **kwargs: Additional agent-specific arguments

    Returns:
        An instance of BaseAgent
    """
    agents = {
        "context": ContextAwareAgent,
        "context_aware": ContextAwareAgent,
        "no_context": NoContextAgent,
        "no-context": NoContextAgent,
        "baseline": NoContextAgent,
    }

    if agent_type.lower() not in agents:
        available = ", ".join(agents.keys())
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")

    agent_class = agents[agent_type.lower()]
    return agent_class(deployment=deployment, endpoint=endpoint, **kwargs)
