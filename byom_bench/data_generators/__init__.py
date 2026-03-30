"""
Data generators for BYOM-Bench.

This module provides strategies for generating conversation data with user preferences.

Available generators:
    - MultiSessionGenerator: Multi-session generation with life-event-driven preference evolution

Usage:
    from byom_bench.data_generators import MultiSessionGenerator

    generator = MultiSessionGenerator(
        persona="Software engineer considering career change...",
        num_sessions=2,
    )
    result = generator.generate_multi_session()  # Returns MultiSessionOutput
"""

from byom_bench.data_generators.multisession import GenerationError, MultiSessionGenerator

__all__ = [
    "GenerationError",
    "MultiSessionGenerator",
]
