"""
Evaluation Task Generator for Multi-Session Preference Recall.

Generates evaluation tasks from multi-session data:
1. Analyzes preference timeline to identify current vs stale preferences
2. Identifies evolved preferences (critical for testing)
3. Randomly selects preferences with round-robin diversity across tasks
4. Creates rubrics for evaluation

No LLM calls — preference selection is purely random.
"""

import random
import uuid
from typing import Any

from memory_gym.client import PIPELINE_CONFIG
from memory_gym.schemas import (
    EvaluationRubric,
    EvaluationTaskSpec,
    MultiSessionOutput,
    Preference,
)

_task_cfg = PIPELINE_CONFIG["task_generation"]


class EvaluationTaskGenerator:
    """Generates evaluation tasks from multi-session conversation data.

    Preference selection is purely random with round-robin diversity:
    evolved preferences are sampled first (minimum enforced), then
    baseline preferences fill the remaining slots. Across tasks in a
    batch, preferences cycle through all available before reuse.

    No LLM calls. No client needed.

    Example usage:
        >>> from memory_gym.task_generators import EvaluationTaskGenerator
        >>> generator = EvaluationTaskGenerator()
        >>> tasks = generator.generate_batch(multisession_output, num_tasks=3)
        >>> for task in tasks:
        ...     print(task.rubric.required_preferences)
    """

    DEFAULT_NUM_TASKS: int = _task_cfg["default_num_tasks"]
    DEFAULT_PREFS_PER_TASK: int = _task_cfg["prefs_per_task"]
    MIN_EVOLVED_PREFS: int = _task_cfg["min_evolved_prefs"]
    EVOLVED_PREF_RATIO: float = _task_cfg["evolved_pref_ratio"]

    def generate_batch(
        self,
        multisession_output: MultiSessionOutput,
        num_tasks: int = DEFAULT_NUM_TASKS,
        prefs_per_task: int = DEFAULT_PREFS_PER_TASK,
    ) -> list[EvaluationTaskSpec]:
        """Generate multiple evaluation tasks with round-robin preference diversity.

        Each task gets a mix of evolved and baseline preferences. Across tasks,
        preferences cycle through all available before any can be reused.

        Args:
            multisession_output: Complete multi-session generation output
            num_tasks: Number of tasks to generate
            prefs_per_task: Number of preferences per task

        Returns:
            List of EvaluationTaskSpec objects
        """
        current_prefs = multisession_output.get_current_preferences()
        evolutions = multisession_output.get_evolved_preferences()
        evolved_ids = multisession_output.get_evolved_preference_ids()

        evolved_prefs = [p for p in current_prefs if p.preference_id in evolved_ids]
        baseline_prefs = [p for p in current_prefs if p.preference_id not in evolved_ids]

        num_evolved_required = max(self.MIN_EVOLVED_PREFS, int(prefs_per_task * self.EVOLVED_PREF_RATIO))
        num_baseline_required = prefs_per_task - num_evolved_required

        if len(evolved_prefs) < num_evolved_required:
            print(
                f"Only {len(evolved_prefs)} evolved preferences available, "
                f"need {num_evolved_required}. Will use all evolved preferences."
            )
            num_evolved_required = len(evolved_prefs)
            num_baseline_required = prefs_per_task - num_evolved_required

        if len(baseline_prefs) < num_baseline_required:
            print(f"Only {len(baseline_prefs)} baseline preferences available, will adjust task preference count.")
            num_baseline_required = len(baseline_prefs)

        # Round-robin pools: shuffle once, cycle through all before reuse
        evolved_pool = list(evolved_prefs)
        baseline_pool = list(baseline_prefs)
        random.shuffle(evolved_pool)
        random.shuffle(baseline_pool)

        evolved_index = 0
        baseline_index = 0

        tasks: list[EvaluationTaskSpec] = []
        for _ in range(num_tasks):
            # Select evolved prefs with round-robin
            selected_evolved: list[Preference] = []
            for _ in range(num_evolved_required):
                if evolved_index >= len(evolved_pool):
                    random.shuffle(evolved_pool)
                    evolved_index = 0
                selected_evolved.append(evolved_pool[evolved_index])
                evolved_index += 1

            # Select baseline prefs with round-robin
            selected_baseline: list[Preference] = []
            for _ in range(num_baseline_required):
                if baseline_index >= len(baseline_pool):
                    random.shuffle(baseline_pool)
                    baseline_index = 0
                selected_baseline.append(baseline_pool[baseline_index])
                baseline_index += 1

            selected_prefs = selected_evolved + selected_baseline

            # Build rubric with supersedes info for evolved prefs
            selected_evolved_ids = {p.preference_id for p in selected_evolved}
            relevant_stale = [old for old, new in evolutions if new.preference_id in selected_evolved_ids]

            rubric = self._build_rubric(
                required_prefs=selected_prefs,
                relevant_stale=relevant_stale,
            )

            task = EvaluationTaskSpec(
                task_id=f"eval_{uuid.uuid4().hex[:8]}",
                rubric=rubric,
                persona=multisession_output.persona,
            )
            tasks.append(task)

        return tasks

    def generate(
        self,
        multisession_output: MultiSessionOutput,
    ) -> EvaluationTaskSpec:
        """Generate a single evaluation task.

        Args:
            multisession_output: Complete multi-session generation output

        Returns:
            EvaluationTaskSpec ready for evaluation
        """
        tasks = self.generate_batch(multisession_output, num_tasks=1)
        return tasks[0]

    def _build_rubric(
        self,
        required_prefs: list[Preference],
        relevant_stale: list[Preference] | None = None,
    ) -> EvaluationRubric:
        """Build the evaluation rubric for the judge.

        The rubric defines required preferences as objects with id, fact,
        and optional supersedes info. The supersedes field is used by the
        judge for stale preference detection.

        Args:
            required_prefs: Preferences that MUST be applied for this task
            relevant_stale: Stale preferences that are counterparts to the
                            required evolved preferences
        """
        relevant_stale = relevant_stale or []

        # Build map of stale preferences by their superseded_by id
        stale_by_superseded: dict[str, Preference] = {}
        for sp in relevant_stale:
            if sp.superseded_by:
                stale_by_superseded[sp.superseded_by] = sp

        # Build required_preferences with optional supersedes info
        required_prefs_objects: list[dict[str, Any]] = []
        for p in required_prefs:
            pref_entry: dict[str, Any] = {
                "id": p.preference_id,
                "fact": p.fact,
            }
            # If this preference supersedes a stale one, include that info
            if p.preference_id in stale_by_superseded:
                stale_pref = stale_by_superseded[p.preference_id]
                pref_entry["supersedes"] = {
                    "id": stale_pref.preference_id,
                    "fact": stale_pref.fact,
                }
            required_prefs_objects.append(pref_entry)

        return EvaluationRubric(
            required_preferences=required_prefs_objects,
        )


def generate_evaluation_tasks(
    multisession_output: MultiSessionOutput,
    num_tasks: int = EvaluationTaskGenerator.DEFAULT_NUM_TASKS,
    prefs_per_task: int = EvaluationTaskGenerator.DEFAULT_PREFS_PER_TASK,
) -> list[EvaluationTaskSpec]:
    """Generate multiple evaluation tasks from multi-session data.

    Pure random selection with round-robin diversity, no LLM calls.
    Each task gets a mix of evolved and baseline preferences.

    Args:
        multisession_output: Output from MultiSessionGenerator
        num_tasks: Number of tasks to generate
        prefs_per_task: Number of preferences required per task

    Returns:
        List of EvaluationTaskSpec objects ready for evaluation
    """
    generator = EvaluationTaskGenerator()
    return generator.generate_batch(multisession_output, num_tasks, prefs_per_task)
