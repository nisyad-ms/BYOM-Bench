"""Pydantic models for structured conversation data in PersonaMem.

This module re-exports models from persona_gym.schemas for backward compatibility.
New code should import directly from persona_gym.schemas.
"""

# Re-export from consolidated schemas module
from persona_gym.schemas import (
    ConversationTurn,
    GeneratedConversation,
    SideNote,
)

__all__ = [
    "SideNote",
    "ConversationTurn",
    "GeneratedConversation",
]
