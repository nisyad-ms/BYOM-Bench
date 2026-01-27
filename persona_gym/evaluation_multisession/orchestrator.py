"""
Evaluation Orchestrator for Multi-Session Preference Recall.

This module provides backward-compatible access to task generation functionality.
The actual implementation has been moved to persona_gym.task_generators.

For direct task generation (without evaluation), use:
    from persona_gym.task_generators import EvaluationTaskGenerator, generate_evaluation_task
"""

import logging

from persona_gym.client import LLMClient
from persona_gym.schemas import (
    EvaluationTask,
    MultiSessionOutput,
)
from persona_gym.task_generators import EvaluationTaskGenerator

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Generates evaluation tasks from multi-session conversation data.

    This class is a thin wrapper around EvaluationTaskGenerator for
    backward compatibility. For new code, consider using
    EvaluationTaskGenerator directly.

    The orchestrator:
    1. Analyzes the preference timeline to identify current vs stale preferences
    2. Generates an evaluation event that will test preference recall
    3. Creates a user prompt that naturally elicits task completion
    4. Builds a rubric for the judge to score the conversation
    """

    def __init__(self, client: LLMClient | None = None):
        """Initialize orchestrator with optional LLM client.

        Args:
            client: LLM client for generation. If None, creates a new one.
        """
        self._generator = EvaluationTaskGenerator(client)

    def generate_evaluation_task(
        self,
        multisession_output: MultiSessionOutput,
        num_stale_traps: int = 2,
    ) -> EvaluationTask:
        """Generate an evaluation task from multi-session data.

        Args:
            multisession_output: Complete multi-session generation output
            num_stale_traps: Number of stale preferences to include as traps

        Returns:
            EvaluationTask ready for evaluation
        """
        return self._generator.generate(multisession_output, num_stale_traps)


def generate_evaluation_from_multisession(
    multisession_output: MultiSessionOutput,
    client: LLMClient | None = None,
    num_stale_traps: int = 2,
) -> EvaluationTask:
    """Convenience function to generate an evaluation task from multi-session data.

    This function is deprecated. Use persona_gym.task_generators.generate_evaluation_task instead.

    Args:
        multisession_output: Output from MultiSessionGenerator
        client: Optional LLM client
        num_stale_traps: Number of stale preferences to use as traps

    Returns:
        EvaluationTask ready for evaluation
    """
    from persona_gym.task_generators import generate_evaluation_task
    return generate_evaluation_task(multisession_output, client, num_stale_traps)
