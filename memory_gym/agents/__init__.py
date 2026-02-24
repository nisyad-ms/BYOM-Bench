"""
Agent implementations for evaluation.

Provides different agent types:
- ContextAwareAgent: Has access to full preference history
- NoContextAgent: No memory of past conversations (baseline)
- FoundryMemoryAgent: Uses Azure AI Foundry memory store
- FoundryLocalAgent: Uses local LanceDB memory (replicates Foundry pipeline)
- GoogleMemoryAgent: Uses Google Vertex AI Agent Engine memory
- AWSMemoryAgent: Uses AWS Bedrock AgentCore memory
"""

from .aws_agent import AWSMemoryAgent
from .context_aware import ContextAwareAgent
from .foundry_agent import FoundryMemoryAgent, get_foundry_configs
from .foundry_local_agent import FoundryLocalAgent
from .google_agent import GoogleMemoryAgent
from .no_context import NoContextAgent

__all__ = [
    "AWSMemoryAgent",
    "ContextAwareAgent",
    "FoundryLocalAgent",
    "FoundryMemoryAgent",
    "get_foundry_configs",
    "GoogleMemoryAgent",
    "NoContextAgent",
]
