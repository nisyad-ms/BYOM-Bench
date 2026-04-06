"""
Agent implementations for evaluation.

Provides different agent types:
- ContextAwareAgent: Has access to full preference history
- NoContextAgent: No memory of past conversations (baseline)
- MemoryAgent: Generic wrapper for any MemoryStore implementation
- MemoryStore: Protocol for pluggable memory backends

Core memory store implementations:
- FoundryMemoryStore: Azure AI Foundry memory store API
- GoogleMemoryStore: Google Vertex AI Agent Engine memory
- AWSMemoryStore: AWS Bedrock AgentCore memory

Optional memory stores are autodiscovered from ``stores/`` when their
dependencies are installed.  See ``BRING_YOUR_OWN_MEMORY.md`` for details.
"""

from .context_aware import ContextAwareAgent
from .memory_agent import MemoryAgent
from .no_context import NoContextAgent
from .stores import (
    _STORE_REGISTRY,
    FoundryMemoryStore,
    MemoryStore,
    get_available_agent_types,
    get_foundry_configs,
    get_store_class,
)

__all__ = [
    "ContextAwareAgent",
    "FoundryMemoryStore",
    "MemoryAgent",
    "MemoryStore",
    "NoContextAgent",
    "get_available_agent_types",
    "get_foundry_configs",
    "get_store_class",
]

# Re-export all autodiscovered store classes so existing
# ``from byom_bench.agents import Mem0MemoryStore`` continues to work.
for _name, _cls in _STORE_REGISTRY.items():
    globals()[_cls.__name__] = _cls
    if _cls.__name__ not in __all__:
        __all__.append(_cls.__name__)
