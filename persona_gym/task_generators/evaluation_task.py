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

from persona_gym.client import LLMClient
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
        ...     print(task.evaluation_event.task)
        ...     print(task.rubric.required_preferences)
    """

    # Default configuration
    DEFAULT_NUM_TASKS = 3
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

        Returns:
            Single EvaluationTask
        """
        # Select preferences for this task (use index for variation)
        random.seed(42 + task_index)  # Reproducible but different per task

        selected_evolved = random.sample(evolved_prefs, min(num_evolved, len(evolved_prefs)))
        selected_baseline = random.sample(baseline_prefs, min(num_baseline, len(baseline_prefs)))
        selected_prefs = selected_evolved + selected_baseline

        # Find stale preferences that correspond to selected evolved ones
        selected_evolved_ids = {p.preference_id for p in selected_evolved}
        relevant_stale = [
            old for old, new in evolutions
            if new.preference_id in selected_evolved_ids
        ]

        logger.debug(
            f"Task {task_index}: {len(selected_evolved)} evolved, "
            f"{len(selected_baseline)} baseline, {len(relevant_stale)} relevant stale traps"
        )

        # Generate evaluation event with full context
        evaluation_event = self._generate_evaluation_event(
            multisession_output=multisession_output,
            required_prefs=selected_prefs,
            required_evolutions=[(old, new) for old, new in evolutions if new.preference_id in selected_evolved_ids],
            num_evolved=len(selected_evolved),
            num_baseline=len(selected_baseline),
        )

        # Build rubric with required preferences and completion criteria
        rubric = self._build_rubric(
            current_prefs=multisession_output.get_current_preferences(),
            stale_prefs=stale_prefs,
            required_prefs=selected_prefs,
            relevant_stale=relevant_stale,
        )

        # Create persona summary for user simulator
        persona_summary = self._create_persona_summary(
            persona=multisession_output.persona,
            current_prefs=selected_prefs,
        )

        return EvaluationTask(
            task_id=f"eval_{uuid.uuid4().hex[:8]}",
            evaluation_event=evaluation_event,
            rubric=rubric,
            persona_summary=persona_summary,
            max_turns=20,  # Hard cap at 20 turns
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

    def _generate_evaluation_event(
        self,
        multisession_output: MultiSessionOutput,
        required_prefs: list[Preference],
        required_evolutions: list[tuple[Preference, Preference]],
        num_evolved: int,
        num_baseline: int,
    ) -> LifeEvent:
        """Generate an evaluation event that tests preference recall.

        Provides the LLM with a chronological narrative of the user's preference
        evolution - how life events shaped their preferences over time.

        Args:
            multisession_output: Full multi-session data with preference timeline
            required_prefs: Preferences that MUST be tested in this task
            required_evolutions: (old, new) pairs for required evolved preferences
            num_evolved: Number of evolved preferences in required set
            num_baseline: Number of baseline preferences in required set
        """
        # Build chronological preference evolution story
        preference_story = self._build_preference_evolution_story(multisession_output)

        # Format REQUIRED preferences for this task
        required_evolved_ids = {new.preference_id for _, new in required_evolutions}

        required_prefs_list = []
        for p in required_prefs:
            pref_info = {"id": p.preference_id, "fact": p.fact, "domain": p.domain}
            if p.preference_id in required_evolved_ids:
                pref_info["is_evolved"] = True
                for old, new in required_evolutions:
                    if new.preference_id == p.preference_id:
                        pref_info["evolved_from"] = old.fact
                        pref_info["evolution_reason"] = old.reason_for_change or ""
                        break
            required_prefs_list.append(pref_info)

        prompt = render_prompt(
            "evaluation/generate_evaluation_event",
            persona=multisession_output.persona,
            preference_evolution_story=preference_story,
            required_preferences=json.dumps(required_prefs_list, indent=2, ensure_ascii=False),
            num_required=len(required_prefs),
            num_evolved=num_evolved,
            num_baseline=num_baseline,
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=render_prompt("evaluation/task_generator_system"),
        )

        try:
            return LifeEvent(
                session_id=-1,  # Evaluation event, not part of training sessions
                date=response.get("date", datetime.now().strftime("%m/%d/%Y")),
                event=response.get("event", "General task"),
                context=response.get("context", ""),
                user_prompt=response.get("user_prompt", ""),  # Natural message (for agent)
                task_internal=response.get("task_internal", ""),  # Detailed task (for judge)
                task=response.get("task", ""),  # Legacy fallback
            )
        except KeyError as e:
            logger.error(f"Failed to parse evaluation event: {e}")
            # Fallback to generic event
            return LifeEvent(
                session_id=-1,
                date=datetime.now().strftime("%m/%d/%Y"),
                event="User needs help with a task",
                context="General assistance request",
            )

    def _build_preference_evolution_story(self, multisession_output: MultiSessionOutput) -> str:
        """Build a chronological narrative of preference evolution.

        Format:
        BASELINE PREFERENCES (before life events):
        - [domain] preference fact

        SESSION 0 (date): Life event description
        ├─ EVOLVED: "old fact" → "new fact"
        ├─ NEW: "new preference created by this event"
        └─ UNCHANGED: 23 other preferences remain active

        ...

        CURRENT STATE SUMMARY:
        Active preferences: X total (Y evolved, Z baseline)
        Stale preferences: W total (superseded by life events)
        """
        lines = []
        timeline = multisession_output.timeline

        # Get all preferences
        all_prefs = list(timeline.preferences.values())
        stale_prefs = [p for p in all_prefs if not p.is_active]
        current_prefs = multisession_output.get_current_preferences()
        evolved_ids = multisession_output.get_evolved_preference_ids()

        # Baseline prefs are those created at session -1
        original_baseline_prefs = [p for p in all_prefs if p.created_at_session == -1]

        # Build lookup for quick access
        pref_by_id: dict[str, Preference] = timeline.preferences

        # BASELINE PREFERENCES (session_id = -1)
        lines.append("BASELINE PREFERENCES (established before any life events):")
        for p in original_baseline_prefs:
            # Check if this was later superseded
            if p.is_active:
                lines.append(f"  - [{p.domain}] {p.fact}")
            else:
                superseding_pref = pref_by_id.get(p.superseded_by or "", None)
                if superseding_pref:
                    lines.append(f"  - [{p.domain}] {p.fact} ⚠️ (later changed)")
                else:
                    lines.append(f"  - [{p.domain}] {p.fact}")
        lines.append("")

        # SESSIONS - chronological events
        for session in multisession_output.sessions:
            event = session.life_event
            lines.append(f"SESSION {session.session_id} ({event.date}): {event.event}")

            # What evolved in this session?
            evolved_pairs: list[tuple[Preference, Preference]] = []
            for old_id, new_id in session.evolved_preference_ids.items():
                old_pref = pref_by_id.get(old_id)
                new_pref = pref_by_id.get(new_id)
                if old_pref and new_pref:
                    evolved_pairs.append((old_pref, new_pref))

            # Get IDs of preferences created by evolution (so we don't double-list them)
            evolved_new_ids = set(session.evolved_preference_ids.values())

            # What was newly created in this session? (exclude evolved ones)
            new_prefs: list[Preference] = [
                pref_by_id[pid] for pid in session.new_preference_ids
                if pid in pref_by_id and pid not in evolved_new_ids
            ]

            # Format evolutions
            for old_p, new_p in evolved_pairs:
                reason = old_p.reason_for_change or "life circumstances changed"
                lines.append(f"  ├─ EVOLVED [{new_p.domain}]: \"{old_p.fact}\"")
                lines.append(f"  │           → \"{new_p.fact}\"")
                lines.append(f"  │           (reason: {reason})")

            # Format new preferences
            for new_p in new_prefs:
                lines.append(f"  ├─ NEW [{new_p.domain}]: \"{new_p.fact}\"")

            lines.append("")

        # CURRENT STATE SUMMARY
        baseline_active = [p for p in current_prefs if p.preference_id not in evolved_ids]

        lines.append("CURRENT STATE SUMMARY:")
        lines.append(f"  Active preferences: {len(current_prefs)} total ({len(evolved_ids)} evolved, {len(baseline_active)} unchanged baseline)")
        lines.append(f"  Stale preferences: {len(stale_prefs)} (superseded by life events)")
        lines.append("")

        # List current preferences
        lines.append("ALL CURRENT PREFERENCES:")
        for p in current_prefs:
            marker = "🔄" if p.preference_id in evolved_ids else "📌"
            lines.append(f"  {marker} [{p.domain}] {p.fact}")

        lines.append("")
        lines.append("ALL STALE PREFERENCES (DO NOT recommend these):")
        for p in stale_prefs:
            superseding = pref_by_id.get(p.superseded_by or "")
            supersede_fact = superseding.fact if superseding else "unknown"
            lines.append(f"  ❌ [{p.domain}] \"{p.fact}\" → replaced by: \"{supersede_fact}\"")

        return "\n".join(lines)

    def _build_rubric(
        self,
        current_prefs: list[Preference],
        stale_prefs: list[Preference],
        required_prefs: list[Preference],
        relevant_stale: list[Preference] | None = None,
        completion_criteria: dict[str, str] | None = None,
    ) -> EvaluationRubric:
        """Build the evaluation rubric for the judge.

        The rubric defines:
        - Current preferences the agent SHOULD use
        - Stale preferences the agent should NOT use (only task-relevant ones)
        - Required preferences that MUST be applied for this task
        - Clear completion criteria
        - Expected good behaviors
        - Trap behaviors (only for stale counterparts of required evolved prefs)

        Args:
            current_prefs: All current active preferences
            stale_prefs: All stale preferences (for reference)
            required_prefs: Preferences that MUST be applied for this task
            relevant_stale: Stale preferences that are counterparts to the
                            required evolved preferences (these become traps)
            completion_criteria: Per-preference criteria for task completion
        """
        required_pref_ids = [p.preference_id for p in required_prefs]
        completion_criteria = completion_criteria or {}

        # Build expected behaviors - prioritize required preferences
        expected_behaviors = []
        for p in required_prefs:
            expected_behaviors.append(
                f"MUST explicitly apply: '{p.fact}' (required for this task)"
            )

        # Add completion criteria
        if completion_criteria:
            expected_behaviors.append(
                f"Task completion: {completion_criteria}"
            )
        else:
            expected_behaviors.append(
                "Complete the user's task without requiring excessive clarification"
            )

        # Build trap behaviors ONLY from relevant_stale
        # These are the stale counterparts of the required evolved preferences
        trap_behaviors = []

        if relevant_stale:
            for sp in relevant_stale:
                new_pref = None
                for cp in current_prefs:
                    if cp.preference_id == sp.superseded_by:
                        new_pref = cp
                        break
                if new_pref:
                    trap_behaviors.append(
                        f"CRITICAL TRAP: Using outdated '{sp.fact}' instead of current '{new_pref.fact}'"
                    )
                else:
                    trap_behaviors.append(
                        f"CRITICAL TRAP: Using superseded preference: '{sp.fact}'"
                    )

        # Only include relevant stale preferences in the rubric, not all of them
        rubric_stale = relevant_stale or []

        return EvaluationRubric(
            current_preferences=current_prefs,
            stale_preferences=rubric_stale,
            expected_behaviors=expected_behaviors,
            trap_behaviors=trap_behaviors,
            required_preferences=required_pref_ids,
            completion_criteria=completion_criteria,
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
        >>> with open("outputs/data_generation_output.json") as f:
        ...     data = MultiSessionOutput.from_dict(json.load(f))
        >>> tasks = generate_evaluation_tasks(data, num_tasks=3)
        >>> for task in tasks:
        ...     print(f"Task: {task.evaluation_event.task}")
        ...     print(f"Required: {len(task.rubric.required_preferences)} preferences")
    """
    generator = EvaluationTaskGenerator(client)
    return generator.generate_batch(multisession_output, num_tasks, prefs_per_task)
