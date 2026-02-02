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

import json
import logging
import random
import uuid
from datetime import datetime

from persona_gym.client import AsyncLLMPool, LLMClient
from persona_gym.prompts import render_prompt
from persona_gym.schemas import (
    EvaluationRubric,
    EvaluationTask,
    LifeEvent,
    MultiSessionOutput,
    Preference,
)

logger = logging.getLogger(__name__)


class EvaluationTaskGenerator:
    """Generates evaluation tasks from multi-session conversation data.

    The generator:
    1. Analyzes the preference timeline to identify current vs stale preferences
    2. Identifies evolved preferences (critical for testing recall)
    3. Generates evaluation events that require preference recall
    4. Creates user prompts and rubrics for evaluation

    Example usage:
        >>> from persona_gym.task_generators import EvaluationTaskGenerator
        >>> generator = EvaluationTaskGenerator()
        >>> tasks = generator.generate_batch(multisession_output, num_tasks=3)
        >>> for task in tasks:
        ...     print(task.evaluation_event.task_internal)
        ...     print(task.rubric.required_preferences)
    """

    # Default configuration
    DEFAULT_NUM_TASKS = 1
    DEFAULT_PREFS_PER_TASK = 6
    MIN_EVOLVED_PREFS = 3  # Minimum evolved preferences per task
    EVOLVED_PREF_RATIO = 0.5  # Target 50% evolved preferences

    def __init__(self, client: LLMClient | None = None):
        """Initialize task generator with optional LLM client.

        Args:
            client: LLM client for generation. If None, creates a new one.
        """
        self.client = client or LLMClient()

    def generate_batch(
        self,
        multisession_output: MultiSessionOutput,
        num_tasks: int = DEFAULT_NUM_TASKS,
        prefs_per_task: int = DEFAULT_PREFS_PER_TASK,
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
        stale_prefs = multisession_output.get_superseded_preferences()
        evolutions = multisession_output.get_evolved_preferences()
        evolved_ids = multisession_output.get_evolved_preference_ids()

        # Separate evolved and baseline current preferences
        evolved_prefs = [p for p in current_prefs if p.preference_id in evolved_ids]
        baseline_prefs = [p for p in current_prefs if p.preference_id not in evolved_ids]

        logger.info(
            f"Preference landscape: {len(current_prefs)} current "
            f"({len(evolved_prefs)} evolved, {len(baseline_prefs)} baseline), "
            f"{len(stale_prefs)} stale"
        )

        # Calculate required evolved preferences per task
        # Target 50% but at least MIN_EVOLVED_PREFS
        num_evolved_required = max(
            self.MIN_EVOLVED_PREFS,
            int(prefs_per_task * self.EVOLVED_PREF_RATIO)
        )
        num_baseline_required = prefs_per_task - num_evolved_required

        # Validate we have enough preferences
        if len(evolved_prefs) < num_evolved_required:
            logger.warning(
                f"Only {len(evolved_prefs)} evolved preferences available, "
                f"need {num_evolved_required}. Will use all evolved preferences."
            )
            num_evolved_required = len(evolved_prefs)
            num_baseline_required = prefs_per_task - num_evolved_required

        if len(baseline_prefs) < num_baseline_required:
            logger.warning(
                f"Only {len(baseline_prefs)} baseline preferences available, "
                f"will adjust task preference count."
            )
            num_baseline_required = len(baseline_prefs)

        logger.info(
            f"Task config: {prefs_per_task} prefs/task "
            f"({num_evolved_required} evolved, {num_baseline_required} baseline)"
        )

        # Generate tasks
        tasks = []
        for i in range(num_tasks):
            logger.info(f"Generating task {i + 1}/{num_tasks}...")
            task = self._generate_single_task(
                multisession_output=multisession_output,
                evolved_prefs=evolved_prefs,
                baseline_prefs=baseline_prefs,
                stale_prefs=stale_prefs,
                evolutions=evolutions,
                num_evolved=num_evolved_required,
                num_baseline=num_baseline_required,
                task_index=i,
            )
            tasks.append(task)

        return tasks

    def _generate_single_task(
        self,
        multisession_output: MultiSessionOutput,
        evolved_prefs: list[Preference],
        baseline_prefs: list[Preference],
        stale_prefs: list[Preference],
        evolutions: list[tuple[Preference, Preference]],
        num_evolved: int,
        num_baseline: int,
        task_index: int,
        use_v2: bool = True,
    ) -> EvaluationTask:
        """Generate a single evaluation task with specified preference mix.

        Args:
            multisession_output: Source data
            evolved_prefs: List of evolved (current) preferences
            baseline_prefs: List of baseline (unchanged) preferences
            stale_prefs: All stale preferences (for traps)
            evolutions: List of (old, new) preference pairs
            num_evolved: Number of evolved preferences to require
            num_baseline: Number of baseline preferences to require
            task_index: Index for shuffling seed
            use_v2: Use v2 prompt where LLM selects preferences

        Returns:
            Single EvaluationTask
        """
        if use_v2:
            evaluation_event, selected_pref_dicts = self._generate_evaluation_event_v2(
                multisession_output=multisession_output,
                num_evolved=num_evolved,
                num_baseline=num_baseline,
            )

            timeline = multisession_output.timeline
            selected_prefs = []
            for pref_dict in selected_pref_dicts:
                pref_id = pref_dict.get("id", "")
                if pref_id in timeline.preferences:
                    selected_prefs.append(timeline.preferences[pref_id])

            selected_evolved_ids = {p.preference_id for p in selected_prefs if p.preference_id in multisession_output.get_evolved_preference_ids()}
            relevant_stale = [
                old for old, new in evolutions
                if new.preference_id in selected_evolved_ids
            ]
        else:
            selected_evolved = random.sample(evolved_prefs, min(num_evolved, len(evolved_prefs)))
            selected_baseline = random.sample(baseline_prefs, min(num_baseline, len(baseline_prefs)))
            selected_prefs = selected_evolved + selected_baseline

            selected_evolved_ids = {p.preference_id for p in selected_evolved}
            relevant_stale = [
                old for old, new in evolutions
                if new.preference_id in selected_evolved_ids
            ]

            evaluation_event = self._generate_evaluation_event(
                multisession_output=multisession_output,
                required_prefs=selected_prefs,
                required_evolutions=[(old, new) for old, new in evolutions if new.preference_id in selected_evolved_ids],
                num_evolved=len(selected_evolved),
                num_baseline=len(selected_baseline),
            )

        logger.debug(
            f"Task {task_index}: {len(selected_prefs)} prefs selected, "
            f"{len(relevant_stale)} relevant stale traps"
        )

        rubric = self._build_rubric(
            current_prefs=multisession_output.get_current_preferences(),
            stale_prefs=stale_prefs,
            required_prefs=selected_prefs,
            relevant_stale=relevant_stale,
        )

        persona_summary = self._create_persona_summary(
            persona=multisession_output.persona,
            current_prefs=selected_prefs,
        )

        return EvaluationTask(
            task_id=f"eval_{uuid.uuid4().hex[:8]}",
            evaluation_event=evaluation_event,
            rubric=rubric,
            persona_summary=persona_summary,
        )

    def generate(
        self,
        multisession_output: MultiSessionOutput,
    ) -> EvaluationTask:
        """Generate a single evaluation task (legacy interface).

        For new code, prefer generate_batch() which generates multiple
        tasks with proper evolved/baseline preference mix.

        Args:
            multisession_output: Complete multi-session generation output

        Returns:
            EvaluationTask ready for evaluation
        """
        tasks = self.generate_batch(multisession_output, num_tasks=1)
        return tasks[0]

    def _generate_evaluation_event_v2(
        self,
        multisession_output: MultiSessionOutput,
        num_evolved: int,
        num_baseline: int,
    ) -> tuple[LifeEvent, list[dict]]:
        """Generate an evaluation event where LLM selects preferences (v2).

        The LLM chooses coherent preferences that work well together for a task,
        rather than us pre-sampling random ones.

        Args:
            multisession_output: Full multi-session data with preference timeline
            num_evolved: Number of evolved preferences LLM should select
            num_baseline: Number of baseline preferences LLM should select

        Returns:
            Tuple of (LifeEvent, list of selected preference dicts)
        """
        preference_story = self._build_preference_evolution_story(multisession_output)

        prompt = render_prompt(
            "task_generation/evaluation_task_instruction_v2",
            persona=multisession_output.persona,
            preference_evolution_story=preference_story,
            num_evolved_required=num_evolved,
            num_baseline_required=num_baseline,
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=render_prompt("task_generation/evaluation_task_system"),
        )

        selected_prefs = response.get("selected_preferences", [])

        life_event = LifeEvent(
            session_id=-1,
            date=response.get("date", datetime.now().strftime("%m/%d/%Y")),
            event=response.get("event", "General task"),
            domain="",
            user_prompt=response.get("user_prompt", ""),
            task_internal=response.get("task_internal", ""),
        )
        return life_event, selected_prefs

    def _generate_evaluation_event(
        self,
        multisession_output: MultiSessionOutput,
        required_prefs: list[Preference],
        required_evolutions: list[tuple[Preference, Preference]],
        num_evolved: int,
        num_baseline: int,
    ) -> LifeEvent:
        """Generate an evaluation event that tests preference recall (v1).

        Provides the LLM with a chronological narrative of the user's preference
        evolution - how life events shaped their preferences over time.

        Args:
            multisession_output: Full multi-session data with preference timeline
            required_prefs: Preferences that MUST be tested in this task
            required_evolutions: (old, new) pairs for required evolved preferences
            num_evolved: Number of evolved preferences in required set
            num_baseline: Number of baseline preferences in required set

        Returns:
            LifeEvent containing the evaluation task
        """
        preference_story = self._build_preference_evolution_story(multisession_output)

        required_evolved_ids = {new.preference_id for _, new in required_evolutions}

        required_prefs_list = []
        for p in required_prefs:
            pref_info = {"id": p.preference_id, "fact": p.fact, "domain": p.domain}
            if p.preference_id in required_evolved_ids:
                pref_info["is_evolved"] = True
                for old, new in required_evolutions:
                    if new.preference_id == p.preference_id:
                        pref_info["evolved_from"] = old.fact
                        break
            required_prefs_list.append(pref_info)

        prompt = render_prompt(
            "task_generation/evaluation_task_instruction",
            persona=multisession_output.persona,
            preference_evolution_story=preference_story,
            required_preferences=json.dumps(required_prefs_list, indent=2, ensure_ascii=False),
            num_required=len(required_prefs),
            num_evolved=num_evolved,
            num_baseline=num_baseline,
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=render_prompt("task_generation/evaluation_task_system"),
        )

        try:
            life_event = LifeEvent(
                session_id=-1,
                date=response.get("date", datetime.now().strftime("%m/%d/%Y")),
                event=response.get("event", "General task"),
                domain=response.get("domain", ""),
                user_prompt=response.get("user_prompt", ""),
                task_internal=response.get("task_internal", ""),
            )
            return life_event
        except KeyError as e:
            logger.error(f"Failed to parse evaluation event: {e}")
            return LifeEvent(
                session_id=-1,
                date=datetime.now().strftime("%m/%d/%Y"),
                event="User needs help with a task",
                domain="",
            )

    def _build_preference_evolution_story(self, multisession_output: MultiSessionOutput) -> str:
        """Build preference history using the same ledger format as the agent.

        Structure:
        1. Core Preferences - baseline preferences grouped by domain
        2. Preference Evolution History - per session changes
        """
        parts = []
        timeline = multisession_output.timeline

        baseline_prefs = [
            p for p in timeline.preferences.values()
            if p.created_at_session == -1
        ]
        if baseline_prefs:
            parts.append("CORE PREFERENCES (before any life events):\n")
            by_domain: dict[str, list[Preference]] = {}
            for pref in baseline_prefs:
                by_domain.setdefault(pref.domain, []).append(pref)

            for domain in sorted(by_domain.keys()):
                parts.append(f"[{domain}]")
                for pref in sorted(by_domain[domain], key=lambda p: p.preference_id):
                    if pref.is_active:
                        parts.append(f"  - [{pref.preference_id}] {pref.fact}")
                    else:
                        parts.append(f"  - [{pref.preference_id}] {pref.fact} [CHANGED in session {pref.superseded_at_session}]")
                parts.append("")

        if multisession_output.sessions:
            parts.append("PREFERENCE EVOLUTION HISTORY:\n")
            for session in multisession_output.sessions:
                event = session.life_event
                parts.append(f"Session {session.session_id}: {event.event}\n")

                non_evolved_new = [
                    pid for pid in session.new_preference_ids
                    if pid not in session.evolved_preference_ids.values()
                ]
                if non_evolved_new:
                    parts.append("  New preferences:")
                    for pref_id in non_evolved_new:
                        pref = timeline.preferences.get(pref_id)
                        if pref:
                            parts.append(f"    - [{pref.preference_id}] [{pref.domain}] {pref.fact}")
                    parts.append("")

                if session.evolved_preference_ids:
                    parts.append("  Evolved preferences:")
                    for old_id, new_id in session.evolved_preference_ids.items():
                        old_pref = timeline.preferences.get(old_id)
                        new_pref = timeline.preferences.get(new_id)
                        if old_pref and new_pref:
                            parts.append(f"    - [{new_id}] EVOLVED from [{old_id}]: \"{old_pref.fact}\"")
                            parts.append(f"      → \"{new_pref.fact}\"")
                    parts.append("")

        return "\n".join(parts)

    def _build_rubric(
        self,
        current_prefs: list[Preference],
        stale_prefs: list[Preference],
        required_prefs: list[Preference],
        relevant_stale: list[Preference] | None = None,
    ) -> EvaluationRubric:
        """Build the evaluation rubric for the judge.

        The rubric defines required preferences as objects with id, fact,
        and optional supersedes info. The supersedes field is used by the
        judge for stale preference detection.

        Args:
            current_prefs: All current active preferences (for reference)
            stale_prefs: All stale preferences (for reference)
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

    def _create_persona_summary(
        self,
        persona: str,
        current_prefs: list[Preference],
    ) -> str:
        """Create a brief persona summary for the user simulator.

        This summary helps the user simulator act consistently without
        needing the full persona history.
        """
        # Extract just key facts, avoid long descriptions that may trigger filters
        # Take first sentence of persona only
        first_sentence = persona.split(".")[0] + "." if "." in persona else persona[:200]

        prefs_summary = "\n".join(f"- {p.fact[:100]}" for p in current_prefs[:4])

        return f"{first_sentence}\n\nKey preferences:\n{prefs_summary}"


def generate_evaluation_task(
    multisession_output: MultiSessionOutput,
    client: LLMClient | None = None,
) -> EvaluationTask:
    """Convenience function to generate a single evaluation task.

    For new code, prefer generate_evaluation_tasks() which generates multiple
    tasks with proper evolved/baseline preference mix.

    Args:
        multisession_output: Output from MultiSessionGenerator
        client: Optional LLM client

    Returns:
        EvaluationTask ready for evaluation
    """
    generator = EvaluationTaskGenerator(client)
    return generator.generate(multisession_output)


def generate_evaluation_tasks(
    multisession_output: MultiSessionOutput,
    num_tasks: int = EvaluationTaskGenerator.DEFAULT_NUM_TASKS,
    prefs_per_task: int = EvaluationTaskGenerator.DEFAULT_PREFS_PER_TASK,
    client: LLMClient | None = None,
) -> list[EvaluationTask]:
    """Generate multiple evaluation tasks from multi-session data.

    Each task requires a mix of evolved and baseline preferences, with
    ~50% evolved preferences (minimum 3) to properly test preference recall.

    Args:
        multisession_output: Output from MultiSessionGenerator
        num_tasks: Number of tasks to generate (default 3)
        prefs_per_task: Number of preferences required per task (default 6)
        client: Optional LLM client

    Returns:
        List of EvaluationTask objects ready for evaluation

    Example:
        >>> from persona_gym.task_generators import generate_evaluation_tasks
        >>> from persona_gym.schemas import MultiSessionOutput
        >>> import json
        >>> with open("outputs/conversation/data_generation_output.json") as f:
        ...     data = MultiSessionOutput.from_dict(json.load(f))
        >>> tasks = generate_evaluation_tasks(data, num_tasks=3)
        >>> for task in tasks:
        ...     print(f"Task: {task.evaluation_event.task_internal}")
        ...     print(f"Required: {len(task.rubric.required_preferences)} preferences")
    """
    generator = EvaluationTaskGenerator(client)
    return generator.generate_batch(multisession_output, num_tasks, prefs_per_task)


def _generate_single_task_with_client(
    client: LLMClient,
    context: dict,
) -> EvaluationTask:
    """Generate a single task using provided client (for parallel execution).

    Args:
        client: LLM client to use
        context: Dict containing multisession_output and task config

    Returns:
        EvaluationTask
    """
    generator = EvaluationTaskGenerator(client)
    multisession_output = context["multisession_output"]
    return generator.generate(multisession_output)


async def generate_evaluation_tasks_parallel(
    multisession_output: MultiSessionOutput,
    num_tasks: int = 3,
) -> list[EvaluationTask]:
    """Generate multiple evaluation tasks in parallel across deployments.

    Args:
        multisession_output: Output from MultiSessionGenerator
        num_tasks: Number of tasks to generate

    Returns:
        List of EvaluationTask objects
    """
    pool = AsyncLLMPool()

    contexts = [{"multisession_output": multisession_output} for _ in range(num_tasks)]

    tasks = await pool.run_parallel(
        items=contexts,
        func=_generate_single_task_with_client,
    )

    return tasks
