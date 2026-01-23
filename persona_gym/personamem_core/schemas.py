"""Pydantic models for structured conversation data in PersonaMem.

This module defines the data models used for representing conversations,
persona information, and artifacts generated during the conversation
generation pipeline.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SideNote(BaseModel):
    """Metadata annotation linking a user turn to a persona fact.

    Side notes capture the implicit information being revealed in a user's
    message, connecting it back to the original persona history. This enables
    tracking which persona facts have been communicated during a conversation.

    Attributes:
        event: Description of the persona event/preference being revealed
        date: Date in MM/DD/YYYY format when this event occurred
    """
    event: str = Field(description="Description of the persona event/preference being revealed")
    date: str = Field(description="Date in MM/DD/YYYY format when this event occurred")


class ConversationTurn(BaseModel):
    """A single turn in the conversation.

    Represents one message in the conversation, either from the user or
    the assistant. User turns may optionally include a side note annotation
    that indicates what persona fact is being revealed.

    Attributes:
        role: The speaker - either "user" or "assistant"
        content: The actual message content
        side_note: Optional annotation for user turns indicating what 
                   preference/fact is being revealed
    """
    role: Literal["user", "assistant"]
    content: str
    side_note: Optional[SideNote] = Field(
        default=None,
        description="Annotation for user turns indicating what preference/fact is being revealed"
    )


class GeneratedConversation(BaseModel):
    """Complete conversation generated from persona history.

    Represents a full multi-turn conversation that has been generated
    based on a persona's history. The conversation is organized by topic
    and time period to support longitudinal memory evaluation.

    Attributes:
        turns: List of conversation turns in chronological order
        topic: Topic of conversation (e.g., travel, therapy, food)
        period: Time period identifier (INIT, WEEK, MONTH, YEAR)
    """
    turns: List[ConversationTurn]
    topic: str = Field(description="Topic of conversation: travel, therapy, food, etc.")
    period: str = Field(default="INIT", description="Time period: INIT, WEEK, MONTH, YEAR")
