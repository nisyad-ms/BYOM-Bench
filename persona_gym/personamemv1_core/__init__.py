"""PersonaMem Core Module

Core data generation modules for PersonaMem benchmark.

Modules:
    - utils: Utility functions (Colors, file operations, data processing)
    - prompts: All prompt templates for LLM queries
    - prepare_data: Data generation functions (persona, topics, conversations)
    - query_llm: LLM query interface (original OpenAI Assistants API version)
    - schemas: Re-exports from persona_gym.schemas for backward compatibility

Usage:
    from persona_gym.personamemv1_core import utils, prompts
    from persona_gym.personamem_core.prepare_data import prepare_persona, prepare_topics
    
    # Preferred: import directly from persona_gym.schemas
    from persona_gym.schemas import ConversationTurn, GeneratedConversation, SideNote
"""

# Re-export schemas from the consolidated location for backward compatibility
from persona_gym.schemas import (
    ConversationTurn,
    GeneratedConversation,
    SideNote,
)

from . import prompts, utils

__all__ = [
    'utils',
    'prompts',
    'ConversationTurn',
    'GeneratedConversation',
    'SideNote',
]
