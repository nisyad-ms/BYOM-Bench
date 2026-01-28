"""PersonaGym: A benchmark for evaluating LLM personalization through multi-turn conversations.

This package provides a 2-stage pipeline:
1. Data Generation - Creating synthetic multi-session conversations with evolving preferences
2. Evaluation - Evaluating agents on their ability to recall and apply user preferences

Main modules:
    - data_generators: MultiSessionGenerator for life-event-driven preference evolution
    - task_generators: Generate evaluation tasks from multi-session data (standalone inspection)
    - evaluation_multisession: Evaluate agents on preference recall across sessions
    - schemas: All data models (MultiSessionOutput, PreferenceTimeline, etc.)
    - client: Shared LLM client

Usage:
    from persona_gym.data_generators import MultiSessionGenerator

    # Generate multi-session data with preference evolution
    generator = MultiSessionGenerator(
        persona="Software engineer considering career change...",
        num_sessions=2,
    )
    result = generator.generate_multi_session()

    # Generate evaluation task (inspect without full evaluation)
    from persona_gym.task_generators import generate_evaluation_task
    task = generate_evaluation_task(result)
    print(task.evaluation_event.task)
    print(task.rubric.required_preferences)

    # Run full evaluation
    from persona_gym.evaluation_multisession import run_evaluation
    eval_result = run_evaluation(result, agent_type="full_context")
"""

__version__ = "0.1.0"

# Don't import submodules here to avoid circular imports
# Users should import directly from submodules:
#   from persona_gym.schemas import ...
#   from persona_gym.data_generators import ...
#   from persona_gym.task_generators import ...

__all__ = ["__version__"]
