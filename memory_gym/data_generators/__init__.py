"""
Data generators for MemoryGym.

This module provides strategies for generating conversation data with user preferences.

Available generators:
    - MultiSessionGenerator: Multi-session generation with life-event-driven preference evolution

Usage:
    from memory_gym.data_generators import MultiSessionGenerator

    generator = MultiSessionGenerator(
        persona="Software engineer considering career change...",
        num_sessions=2,
    )
    result = generator.generate_multi_session()  # Returns MultiSessionOutput
"""

from memory_gym.data_generators.multisession import GenerationError, MultiSessionGenerator

__all__ = [
    "GenerationError",
    "MultiSessionGenerator",
]
