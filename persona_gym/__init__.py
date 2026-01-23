"""PersonaGym: A benchmark for evaluating LLM personalization through multi-turn conversations.

This package provides tools for:
1. Data Generation - Creating synthetic user-assistant conversations grounded in persona histories
2. TOD Task Generation - Generating task-oriented dialogue tasks from conversation data
3. Agent Evaluation - Evaluating agents on their ability to recall and apply user preferences

Main modules:
    - data_generation: Entry point for generating conversation data
    - task_generator: Generate TOD tasks from conversations
    - evaluation: Evaluate agents on TOD tasks
    - metric: Evaluation metrics and scoring
    - agent: Agent implementations (ContextAware, NoContext)
    - tool_simulator: Simulated tool responses for evaluation

Usage:
    # Generate conversation data
    from persona_gym.data_generation import generate_sample
    
    # Generate TOD tasks
    from persona_gym.task_generator import generate_tod_tasks
    
    # Evaluate agents
    from persona_gym.evaluation import run_evaluation
"""

__version__ = "0.1.0"

from persona_gym.personamem_core import prompts, schemas, utils

__all__ = [
    "__version__",
    "utils",
    "prompts",
    "schemas",
]
