"""PersonaMem Core Module

Core data generation modules for PersonaMem benchmark.

Modules:
    - utils: Utility functions (Colors, file operations, data processing)
    - prompts: All prompt templates for LLM queries
    - prepare_data: Data generation functions (persona, topics, conversations)
    - query_llm: LLM query interface (original OpenAI Assistants API version)
    - schemas: Pydantic models for structured conversation data

Usage:
    from persona_gym.personamem_core import utils, prompts, schemas
    from persona_gym.personamem_core.prepare_data import prepare_persona, prepare_topics
    from persona_gym.personamem_core.schemas import (
        ConversationTurn, GeneratedConversation, SideNote
    )
"""

from . import prompts, schemas, utils

__all__ = [
    'utils',
    'prompts',
    'schemas',
]
