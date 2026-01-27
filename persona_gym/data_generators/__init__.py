"""
Data generators for PersonaGym.

This module provides different strategies for generating conversation data
with user preferences. All generators produce DataGenerationOutput which
can be consumed by the task generation stage.

Available generators:
    - PersonaMemV2Generator: Token-budgeted generation with preference evolution
    - MultiSessionGenerator: Multi-session generation with life-event-driven preference evolution
    - (Future) RealDataGenerator: Load from existing conversation datasets
    - (Future) TemplateGenerator: Simple template-based generation

Usage:
    from persona_gym.data_generators import PersonaMemV2Generator

    # V2: Token-budgeted generation with fixed preferences
    generator = PersonaMemV2Generator(
        topic="travel",
        token_budget=8000,
        num_preferences=5,
    )
    output = generator.generate()

    # Multi-session: Life-event driven preference evolution
    from persona_gym.data_generators import MultiSessionGenerator
    generator = MultiSessionGenerator(
        persona="Software engineer considering career change...",
        num_sessions=2,
        num_preferences=5,
    )
    result = generator.generate()  # Returns MultiSessionOutput
"""

from persona_gym.data_generators.base import BaseDataGenerator
from persona_gym.data_generators.multisession import MultiSessionGenerator
from persona_gym.data_generators.personamem_v2 import PersonaMemV2Generator
from persona_gym.schemas import DataGenerationMetadata, DataGenerationOutput

__all__ = [
    "BaseDataGenerator",
    "MultiSessionGenerator",
    "PersonaMemV2Generator",
    "DataGenerationOutput",
    "DataGenerationMetadata",
]

