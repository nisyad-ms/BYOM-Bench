"""
Evaluation Task Generator for Multi-Session Preference Recall.

Generates evaluation tasks from multi-session data:
1. Analyzes preference timeline to identify current vs stale preferences
2. Generates an evaluation event that tests preference recall
3. Creates user prompt and rubric for evaluation

This module can be used standalone to generate and inspect evaluation tasks
without running the full evaluation pipeline.
"""

import json
import logging
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
    2. Generates an evaluation event that will test preference recall
    3. Creates a user prompt that naturally elicits task completion
    4. Builds a rubric for the judge to score the conversation

    Example usage:
        >>> from persona_gym.task_generators import EvaluationTaskGenerator
        >>> generator = EvaluationTaskGenerator()
        >>> task = generator.generate(multisession_output)
        >>> print(task.evaluation_event.task)
        >>> print(task.user_prompt)
        >>> print(task.rubric.required_preferences)
    """

    def __init__(self, client: LLMClient | None = None):
        """Initialize task generator with optional LLM client.

        Args:
            client: LLM client for generation. If None, creates a new one.
        """
        self.client = client or LLMClient()

    def generate(
        self,
        multisession_output: MultiSessionOutput,
    ) -> EvaluationTask:
        """Generate an evaluation task from multi-session data.

        Args:
            multisession_output: Complete multi-session generation output

        Returns:
            EvaluationTask ready for evaluation
        """
        # Get current and stale preferences
        current_prefs = multisession_output.get_current_preferences()
        stale_prefs = multisession_output.get_superseded_preferences()

        # Use all stale preferences as traps
        stale_traps = stale_prefs

        logger.info(
            f"Found {len(current_prefs)} current preferences, "
            f"{len(stale_prefs)} stale preferences, using {len(stale_traps)} as traps"
        )

        # Generate evaluation event
        evaluation_event = self._generate_evaluation_event(
            multisession_output.persona,
            multisession_output.life_events,
            current_prefs,
            stale_traps,
        )

        # Generate user prompt
        user_prompt = self._generate_user_prompt(
            multisession_output.persona,
            evaluation_event,
            current_prefs,
        )

        # Build rubric
        rubric = self._build_rubric(current_prefs, stale_traps, evaluation_event)

        # Create persona summary for user simulator
        persona_summary = self._create_persona_summary(
            multisession_output.persona,
            current_prefs,
        )

        return EvaluationTask(
            task_id=f"eval_{uuid.uuid4().hex[:8]}",
            evaluation_event=evaluation_event,
            user_prompt=user_prompt,
            rubric=rubric,
            persona_summary=persona_summary,
            max_turns=10,
        )

    def _generate_evaluation_event(
        self,
        persona: str,
        life_events: list[LifeEvent],
        current_prefs: list[Preference],
        stale_prefs: list[Preference],
    ) -> LifeEvent:
        """Generate an evaluation event that tests preference recall.

        The event should be realistic for the persona and naturally require
        the agent to consider multiple preferences, including opportunities
        to incorrectly use stale ones.
        """
        # Format life events summary
        life_events_summary = "\n".join(
            f"- {e.date}: {e.event}" for e in life_events
        )

        # Format preferences for prompt - include IDs for required_preferences reference
        current_prefs_str = json.dumps(
            [{"id": p.preference_id, "fact": p.fact, "category": p.category} for p in current_prefs],
            indent=2,
            ensure_ascii=False,
        )

        prompt = render_prompt(
            "evaluation/generate_evaluation_event",
            persona=persona,
            life_events_summary=life_events_summary,
            current_preferences=current_prefs_str,
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=render_prompt("evaluation/task_generator_system"),
        )

        try:
            # Parse required_preferences - could be list of IDs or empty
            required_prefs = response.get("required_preferences", [])
            if not isinstance(required_prefs, list):
                required_prefs = []

            return LifeEvent(
                session_id=-1,  # Evaluation event, not part of training sessions
                date=response.get("date", datetime.now().strftime("%m/%d/%Y")),
                event=response.get("event", "General task"),
                context=response.get("context", ""),
                task=response.get("task", ""),
                required_preferences=required_prefs,
                completion_criteria=response.get("completion_criteria", ""),
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

    def _generate_user_prompt(
        self,
        persona: str,
        evaluation_event: LifeEvent,
        current_prefs: list[Preference],
    ) -> str:
        """Generate the initial user message that starts the evaluation dialogue.

        The prompt should be natural and not explicitly mention preferences,
        but the task should allow for preference-based recommendations.
        """
        prefs_str = json.dumps(
            [{"fact": p.fact, "category": p.category} for p in current_prefs],
            indent=2,
            ensure_ascii=False,
        )

        # Build comprehensive event description including task and completion criteria
        event_parts = [f"Event: {evaluation_event.event}"]
        if evaluation_event.task:
            event_parts.append(f"Task: {evaluation_event.task}")
        if evaluation_event.completion_criteria:
            event_parts.append(f"Completion Criteria: {evaluation_event.completion_criteria}")
        if evaluation_event.context:
            event_parts.append(f"Context: {evaluation_event.context}")

        event_description = "\n".join(event_parts)

        prompt = render_prompt(
            "evaluation/generate_user_prompt",
            persona=persona,
            evaluation_event=event_description,
            current_preferences=prefs_str,
        )

        response = self.client.complete(
            prompt=prompt,
            system_prompt=render_prompt("evaluation/user_prompt_generator_system"),
        )

        # Clean up the response - extract just the message
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        return response

    def _build_rubric(
        self,
        current_prefs: list[Preference],
        stale_prefs: list[Preference],
        evaluation_event: LifeEvent,
    ) -> EvaluationRubric:
        """Build the evaluation rubric for the judge.

        The rubric defines:
        - Current preferences the agent SHOULD use
        - Stale preferences the agent should NOT use
        - Required preferences that MUST be applied for this task
        - Clear completion criteria
        - Expected good behaviors
        - Trap behaviors (using stale preferences)
        """
        # Get required preferences from evaluation event
        required_pref_ids = evaluation_event.required_preferences or []

        # Build expected behaviors - prioritize required preferences
        expected_behaviors = []
        for p in current_prefs:
            if p.preference_id in required_pref_ids:
                expected_behaviors.append(
                    f"MUST explicitly apply: '{p.fact}' (required for this task)"
                )
            else:
                expected_behaviors.append(
                    f"Should reference if relevant: '{p.fact}'"
                )

        # Add completion criteria
        if evaluation_event.completion_criteria:
            expected_behaviors.append(
                f"Task completion: {evaluation_event.completion_criteria}"
            )
        else:
            expected_behaviors.append(
                "Complete the user's task without requiring excessive clarification"
            )

        # Build trap behaviors (things that would be wrong)
        trap_behaviors = []
        for sp in stale_prefs:
            # Find what superseded it
            new_pref = None
            for cp in current_prefs:
                if cp.preference_id == sp.superseded_by:
                    new_pref = cp
                    break

            if new_pref:
                trap_behaviors.append(
                    f"Using '{sp.fact}' instead of '{new_pref.fact}'"
                )
            else:
                trap_behaviors.append(
                    f"Using superseded preference: '{sp.fact}'"
                )

        return EvaluationRubric(
            current_preferences=current_prefs,
            stale_preferences=stale_prefs,
            expected_behaviors=expected_behaviors,
            trap_behaviors=trap_behaviors,
            required_preferences=required_pref_ids,
            completion_criteria=evaluation_event.completion_criteria,
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
    """Convenience function to generate an evaluation task from multi-session data.

    Args:
        multisession_output: Output from MultiSessionGenerator
        client: Optional LLM client

    Returns:
        EvaluationTask ready for evaluation

    Example:
        >>> from persona_gym.task_generators import generate_evaluation_task
        >>> from persona_gym.schemas import MultiSessionOutput
        >>> data = MultiSessionOutput.from_json_file("outputs/test_multisession_output.json")
        >>> task = generate_evaluation_task(data)
        >>> print(f"Task: {task.evaluation_event.task}")
        >>> print(f"Required preferences: {task.rubric.required_preferences}")
    """
    generator = EvaluationTaskGenerator(client)
    return generator.generate(multisession_output)
