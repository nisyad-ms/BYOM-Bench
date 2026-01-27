"""PersonaGym: A benchmark for evaluating LLM personalization through multi-turn conversations.

This package provides a 3-stage pipeline:
1. Data Generation - Creating synthetic user-assistant conversations with preferences
2. Task Generation - Generating task-oriented dialogue tasks from conversation data
3. Evaluation - Evaluating agents on their ability to recall and apply user preferences

Pipeline Contracts (for swapping components):
    - DataGenerationOutput: Stage 1 → Stage 2
    - TaskGenerationOutput: Stage 2 → Stage 3
    - EvaluationOutput: Stage 3 → Final Results

Main modules:
    - data_generators: Data generation strategies (PersonaMemGenerator, etc.)
    - task_generator: Generate TOD tasks from conversations
    - evaluation: Evaluate agents on TOD tasks
    - schemas: All data models and pipeline contracts
    - client: Shared LLM client
    - agent: Agent implementations (ContextAware, NoContext)

Usage:
    # Generate data using PersonaMem
    from persona_gym.data_generators import PersonaMemGenerator
    generator = PersonaMemGenerator(topic="travel")
    data_output = generator.generate()

    # Or load from existing files
    from persona_gym.data_generators import PersonaMemGenerator
    data_output = PersonaMemGenerator.from_files("conversation.json", "artifacts.json")

    # Generate TOD tasks
    from persona_gym.task_generator import generate_tod_tasks
    tasks = generate_tod_tasks(data_output)

    # Evaluate agents
    from persona_gym.evaluation import run_evaluation
    results = run_evaluation(data_output, tasks, agent_type="context")

    # Import contracts directly
    from persona_gym.schemas import DataGenerationOutput, TaskGenerationOutput, EvaluationOutput
"""

__version__ = "0.1.0"

# Don't import submodules here to avoid circular imports
# Users should import directly from submodules:
#   from persona_gym.schemas import ...
#   from persona_gym.data_generators import ...

__all__ = ["__version__"]
