"""
Data generators for PersonaGym.

This module provides strategies for generating conversation data with user preferences.

Available generators:
    - MultiSessionGenerator: Multi-session generation with life-event-driven preference evolution

Usage:
    from memory_gym.data_generators import MultiSessionGenerator

    generator = MultiSessionGenerator(
        persona="Software engineer considering career change...",
        num_sessions=2,
    )
    result = generator.generate()  # Returns MultiSessionOutput
"""

from memory_gym.data_generators.base import BaseDataGenerator
from memory_gym.data_generators.multisession import MultiSessionGenerator
from memory_gym.schemas import DataGenerationMetadata, DataGenerationOutput

__all__ = [
    "BaseDataGenerator",
    "MultiSessionGenerator",
    "DataGenerationOutput",
    "DataGenerationMetadata",
]
