"""
Evaluation Task Generator for Multi-Session Preference Recall.

Generates evaluation tasks from multi-session data:
1. Analyzes preference timeline to identify current vs stale preferences
2. Identifies evolved preferences (critical for testing)
3. Generates evaluation events that require preference recall
4. Creates user prompts and rubrics for evaluation

This module can be used standalone to generate and inspect evaluation tasks
without running the full evaluation pipeline.
"""

import uuid
from datetime import datetime

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.formatting import format_preference_history, summarize_events
from memory_gym.prompts import render_prompt
from memory_gym.schemas import (
    EvaluationRubric,
    EvaluationTask,
    LifeEvent,
    MultiSessionOutput,
    Preference,
)


class EvaluationTaskGenerator:
    """Generates evaluation tasks from multi-session conversation data.

    The generator:
    1. Analyzes the preference timeline to identify current vs stale preferences
    2. Identifies evolved preferences (critical for testing recall)
    3. Generates evaluation events that require preference recall
    4. Creates user prompts and rubrics for evaluation

    Example usage:
        >>> from memory_gym.task_generators import EvaluationTaskGenerator
        >>> generator = EvaluationTaskGenerator()
        >>> tasks = generator.generate_batch(multisession_output, num_tasks=3)
        >>> for task in tasks:
        ...     print(task.evaluation_event.event)
        ...     print(task.rubric.required_preferences)
    """

    # Default configuration
    DEFAULT_NUM_TASKS = 1
    DEFAULT_PREFS_PER_TASK = 6
    MIN_EVOLVED_PREFS = 3  # Minimum evolved preferences per task
    EVOLVED_PREF_RATIO = 0.5  # Target 50% evolved preferences

    def __init__(self, client: LLMClient | PooledLLMClient | None = None):
        """Initialize task generator with optional LLM client.

        Args:
            client: LLM client for generation. If None, creates a new one.
        """
        self.client = client or PooledLLMClient()

    def generate_batch(
        self,
        multisession_output: MultiSessionOutput,
        num_tasks: int = DEFAULT_NUM_TASKS,
        prefs_per_task: int = DEFAULT_PREFS_PER_TASK,
        previous_events: list[str] | None = None,
    ) -> list[EvaluationTask]:
        """Generate multiple evaluation tasks from multi-session data.

        Each task requires a mix of evolved and baseline preferences, with
        evolved preferences being critical for testing preference recall.

        Args:
            multisession_output: Complete multi-session generation output
            num_tasks: Number of tasks to generate (default 3)
            prefs_per_task: Number of preferences required per task (default 5)

        Returns:
            List of EvaluationTask objects ready for evaluation
        """
        # Analyze preference landscape
        current_prefs = multisession_output.get_current_preferences()
        evolutions = multisession_output.get_evolved_preferences()
        evolved_ids = multisession_output.get_evolved_preference_ids()

        # Separate evolved and baseline current preferences
        evolved_prefs = [p for p in current_prefs if p.preference_id in evolved_ids]
        baseline_prefs = [p for p in current_prefs if p.preference_id not in evolved_ids]

        # Calculate required evolved preferences per task
        # Target 50% but at least MIN_EVOLVED_PREFS
        num_evolved_required = max(self.MIN_EVOLVED_PREFS, int(prefs_per_task * self.EVOLVED_PREF_RATIO))
        num_baseline_required = prefs_per_task - num_evolved_required

        # Validate we have enough preferences
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

        # Generate tasks
        tasks = []
        generated_events = list(previous_events) if previous_events else []
        used_baseline_prefs: list[str] = []
        for _ in range(num_tasks):
            task = self._generate_single_task(
                multisession_output=multisession_output,
                evolutions=evolutions,
                num_evolved=num_evolved_required,
                num_baseline=num_baseline_required,
                previous_events=generated_events,
                used_baseline_prefs=used_baseline_prefs,
            )
            tasks.append(task)
            generated_events.append(task.evaluation_event.event)
            for pref in task.rubric.required_preferences:
                if not pref.get("supersedes"):
                    used_baseline_prefs.append(pref["id"])

        return tasks

    def _generate_single_task(
        self,
        multisession_output: MultiSessionOutput,
        evolutions: list[tuple[Preference, Preference]],
        num_evolved: int,
        num_baseline: int,
        previous_events: list[str] | None = None,
        used_baseline_prefs: list[str] | None = None,
    ) -> EvaluationTask:
        """Generate a single evaluation task with specified preference mix.

        Args:
            multisession_output: Source data
            evolutions: List of (old, new) preference pairs
            num_evolved: Number of evolved preferences to require
            num_baseline: Number of baseline preferences to require
            previous_events: Previously generated event descriptions
            used_baseline_prefs: Baseline preference IDs already used

        Returns:
            Single EvaluationTask
        """
        evaluation_event, selected_pref_dicts, scenario_type, reasoning = self._generate_evaluation_event(
            multisession_output=multisession_output,
            num_evolved=num_evolved,
            num_baseline=num_baseline,
            previous_events=previous_events,
            used_baseline_prefs=used_baseline_prefs,
        )

        timeline = multisession_output.timeline
        selected_prefs = []
        for pref_dict in selected_pref_dicts:
            pref_id = pref_dict.get("id", "")
            if pref_id in timeline.preferences:
                selected_prefs.append(timeline.preferences[pref_id])

        selected_evolved_ids = {
            p.preference_id
            for p in selected_prefs
            if p.preference_id in multisession_output.get_evolved_preference_ids()
        }
        relevant_stale = [old for old, new in evolutions if new.preference_id in selected_evolved_ids]

        rubric = self._build_rubric(
            required_prefs=selected_prefs,
            relevant_stale=relevant_stale,
        )

        return EvaluationTask(
            task_id=f"eval_{uuid.uuid4().hex[:8]}",
            evaluation_event=evaluation_event,
            rubric=rubric,
            persona_summary=multisession_output.persona,
            scenario_type=scenario_type,
            reasoning=reasoning,
        )

    def generate(
        self,
        multisession_output: MultiSessionOutput,
    ) -> EvaluationTask:
        """Generate a single evaluation task.

        Args:
            multisession_output: Complete multi-session generation output

        Returns:
            EvaluationTask ready for evaluation
        """
        tasks = self.generate_batch(multisession_output, num_tasks=1)
        return tasks[0]

    def _generate_evaluation_event(
        self,
        multisession_output: MultiSessionOutput,
        num_evolved: int,
        num_baseline: int,
        previous_events: list[str] | None = None,
        used_baseline_prefs: list[str] | None = None,
    ) -> tuple[LifeEvent, list[dict], str, str]:
        """Generate an evaluation event where LLM selects preferences.

        The LLM chooses coherent preferences that work well together for a task,
        rather than us pre-sampling random ones.

        Args:
            multisession_output: Full multi-session data with preference timeline
            num_evolved: Number of evolved preferences LLM should select
            num_baseline: Number of baseline preferences LLM should select
            previous_events: Previously generated event descriptions
            used_baseline_prefs: Baseline preference IDs already used in previous tasks

        Returns:
            Tuple of (LifeEvent, list of selected preference dicts, scenario_type, reasoning)
        """
        event_summaries = summarize_events(multisession_output, self.client)
        preference_story = format_preference_history(multisession_output, event_summaries=event_summaries)

        previous_situations_str = ""
        if previous_events:
            previous_situations_str = "\n".join(f"- {e}" for e in previous_events)
        else:
            previous_situations_str = "(none yet)"

        used_baseline_str = ""
        if used_baseline_prefs:
            used_baseline_str = ", ".join(used_baseline_prefs)
        else:
            used_baseline_str = "(none yet)"

        prompt = render_prompt(
            "task_generation/evaluation_task_instruction",
            persona=multisession_output.persona,
            preference_evolution_story=preference_story,
            num_evolved_required=num_evolved,
            num_baseline_required=num_baseline,
            previous_situations=previous_situations_str,
            used_baseline_preferences=used_baseline_str,
        )

        response = self.client.complete_json(
            prompt=prompt,
            max_tokens=CONFIG["max_tokens"]["task_generation"],
        )

        selected_prefs = response.get("selected_preferences", [])
        situation = response.get("situation", response.get("event", "General task"))
        scenario_type = response.get("scenario_type", "")
        reasoning = response.get("reasoning", response.get("ideal_response", ""))

        user_prompt = self._generate_user_prompt(
            multisession_output=multisession_output,
            scenario_type=scenario_type,
            situation=situation,
            selected_prefs=selected_prefs,
        )

        life_event = LifeEvent(
            session_id=-1,
            date=response.get("date", datetime.now().strftime("%m/%d/%Y")),
            event=situation,
            domain="",
            user_prompt=user_prompt,
        )
        return life_event, selected_prefs, scenario_type, reasoning

    def _generate_user_prompt(
        self,
        multisession_output: MultiSessionOutput,
        scenario_type: str,
        situation: str,
        selected_prefs: list[dict],
    ) -> str:
        """Generate a high-quality opening message for the evaluation conversation.

        This is a separate LLM call focused on creating a directed user prompt
        that tests preferences without revealing them.

        Args:
            multisession_output: Source data for persona summary
            scenario_type: Broad scenario category (e.g., "Daily/weekly routine optimization")
            situation: Detailed situation description (internal reference for the LLM)
            selected_prefs: List of preference dicts being tested

        Returns:
            User's opening message
        """
        persona_summary = multisession_output.persona

        prefs_formatted = "\n".join(f"- [{p.get('id', 'unknown')}] {p.get('fact', '')}" for p in selected_prefs)

        system_prompt = render_prompt(
            "task_generation/user_prompt_generator_system",
            persona_summary=persona_summary,
            scenario_type=scenario_type,
            situation=situation,
            required_preferences=prefs_formatted,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate the user's opening message."},
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"].get("user_prompt_generator", 256),
        )

        return response.strip()

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
        required_prefs_objects: list[dict] = []
        for p in required_prefs:
            pref_entry = {
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
    client: LLMClient | PooledLLMClient | None = None,
    previous_events: list[str] | None = None,
) -> list[EvaluationTask]:
    """Generate multiple evaluation tasks from multi-session data.

    Each task requires a mix of evolved and baseline preferences, with
    ~50% evolved preferences (minimum 3) to properly test preference recall.

    Args:
        multisession_output: Output from MultiSessionGenerator
        num_tasks: Number of tasks to generate (default 3)
        prefs_per_task: Number of preferences required per task (default 6)
        client: Optional LLM client
        previous_events: List of already-generated event descriptions to avoid

    Returns:
        List of EvaluationTask objects ready for evaluation
    """
    generator = EvaluationTaskGenerator(client)
    return generator.generate_batch(multisession_output, num_tasks, prefs_per_task, previous_events)
