"""
Multi-Session Evaluation Module

Evaluates agent preference recall across multiple sessions with preference evolution.

For standalone task generation, use persona_gym.task_generators.
"""

from ream_bench.evaluation_multisession.judge import MultiSessionJudge
from ream_bench.evaluation_multisession.runner import (
    run_evaluation,
    run_evaluations_parallel,
)
from ream_bench.evaluation_multisession.user_simulator import MultiSessionUserSimulator

__all__ = [
    "MultiSessionJudge",
    "MultiSessionUserSimulator",
    "run_evaluation",
    "run_evaluations_parallel",
]
