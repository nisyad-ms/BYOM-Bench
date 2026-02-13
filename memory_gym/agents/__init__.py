"""
Agent implementations for evaluation.

Provides different agent types:
- ContextAwareAgent: Has access to full preference history
- NoContextAgent: No memory of past conversations (baseline)
- FoundryMemoryAgent: Uses Azure AI Foundry memory store
- GoogleMemoryAgent: Uses Google Vertex AI Agent Engine memory
"""

from .base import BaseAgent
from .context_aware import ContextAwareAgent
from .foundry_agent import FoundryMemoryAgent, FoundryMemoryAPIAgent, get_foundry_embedding_models
from .google_agent import GoogleMemoryAgent
from .no_context import NoContextAgent

__all__ = [
    "BaseAgent",
    "ContextAwareAgent",
    "FoundryMemoryAgent",
    "FoundryMemoryAPIAgent",
    "get_foundry_embedding_models",
    "GoogleMemoryAgent",
    "NoContextAgent",
]
