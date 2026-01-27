"""
Data generators for PersonaGym.

This module provides strategies for generating conversation data with user preferences.

Available generators:
    - MultiSessionGenerator: Multi-session generation with life-event-driven preference evolution

Usage:
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
from persona_gym.schemas import DataGenerationMetadata, DataGenerationOutput

__all__ = [
    "BaseDataGenerator",
    "MultiSessionGenerator",
    "DataGenerationOutput",
    "DataGenerationMetadata",
]
