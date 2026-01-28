"""
Multi-Session Judge for Evaluation.

The judge evaluates a completed dialogue by analyzing:
1. How well the agent used current preferences (proactive vs prompted vs ignored)
2. Whether the agent incorrectly used stale preferences
3. Turn efficiency (how many corrections were needed)
4. Task completion
"""

import json
import logging
from typing import Any

import yaml

from persona_gym.client import LLMClient
from persona_gym.prompts import render_prompt
from persona_gym.schemas import (
    EvaluationRubric,
    EvaluationTask,
    LifeEvent,
    MultiSessionEvaluationResult,
)

logger = logging.getLogger(__name__)


class MultiSessionJudge:
    """Evaluates agent performance on multi-session preference recall.

    The judge:
    1. Receives the completed dialogue and rubric
    2. Analyzes each turn for preference usage and corrections
    3. Identifies any stale preference usage
    4. Computes scores based on the rubric
    """

    def __init__(self, client: LLMClient | None = None):
        """Initialize judge with optional LLM client.

        Args:
            client: LLM client for evaluation. If None, creates a new one.
        """
        self.client = client or LLMClient()
        self._few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples from YAML file."""
        try:
            import importlib.resources as pkg_resources

            # Read the YAML file
            prompts_path = pkg_resources.files("persona_gym.prompts.evaluation")
            examples_file = prompts_path / "multisession_judge_examples.yaml"
            content = examples_file.read_text()

            data = yaml.safe_load(content)
            examples = data.get("examples", [])

            # Format examples for the prompt
            formatted = []
            for i, ex in enumerate(examples, 1):
                formatted.append(
                    f"""### Example {i}: {ex['scenario']}

**Current Preferences:**
{json.dumps(ex['current_preferences'], indent=2, ensure_ascii=False)}

**Stale Preferences:**
{json.dumps(ex['stale_preferences'], indent=2, ensure_ascii=False)}

**Evaluation Event:** {ex['evaluation_event']}

**Transcript:**
{json.dumps(ex['transcript'], indent=2, ensure_ascii=False)}

**Correct Evaluation:**
{json.dumps(ex['evaluation'], indent=2, ensure_ascii=False)}
"""
                )

            return "\n---\n".join(formatted)

        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
            return "No few-shot examples available."

    def evaluate(
        self,
        evaluation_task: EvaluationTask,
        conversation: list[dict[str, str]],
    ) -> MultiSessionEvaluationResult:
        """Evaluate a completed dialogue.

        Args:
            evaluation_task: The evaluation task with rubric
            conversation: The dialogue transcript

        Returns:
            MultiSessionEvaluationResult with scores and analysis
        """
        rubric = evaluation_task.rubric

        # Format inputs for judge prompt
        current_prefs_json = json.dumps(
            [
                {
                    "id": p.preference_id,
                    "fact": p.fact,
                    "domain": p.domain,
                }
                for p in rubric.current_preferences
            ],
            indent=2,
            ensure_ascii=False,
        )

        stale_prefs_json = json.dumps(
            [
                {
                    "id": p.preference_id,
                    "fact": p.fact,
                    "superseded_by": p.superseded_by,
                    "reason": p.reason_for_change,
                }
                for p in rubric.stale_preferences
            ],
            indent=2,
            ensure_ascii=False,
        )

        # Format required preferences - include the full preference info for context
        required_prefs_info = []
        for pref_id in rubric.required_preferences:
            for p in rubric.current_preferences:
                if p.preference_id == pref_id:
                    required_prefs_info.append({
                        "id": p.preference_id,
                        "fact": p.fact,
                    })
                    break
        required_prefs_json = json.dumps(required_prefs_info, indent=2, ensure_ascii=False)

        # Get completion criteria
        completion_criteria = rubric.completion_criteria or "Complete the user's task successfully"

        transcript_json = json.dumps(conversation, indent=2, ensure_ascii=False)

        # Build event description with task info
        event_info = evaluation_task.evaluation_event
        event_description = event_info.event
        if event_info.task:
            event_description += f"\n\nSpecific Task: {event_info.task}"

        # Build prompts
        system_prompt = render_prompt("evaluation/multisession_judge_system")

        user_prompt = render_prompt(
            "evaluation/multisession_judge_user",
            few_shot_examples=self._few_shot_examples,
            current_preferences=current_prefs_json,
            stale_preferences=stale_prefs_json,
            required_preferences=required_prefs_json,
            completion_criteria=completion_criteria,
            evaluation_event=event_description,
            transcript=transcript_json,
        )

        # Get judge's evaluation
        try:
            result = self.client.complete_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.1,  # Low temperature for consistent evaluation
            )

            return self._parse_judge_result(
                evaluation_task.task_id,
                result,
                conversation,
                evaluation_task.evaluation_event,
                evaluation_task.rubric,
            )

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            # Return a failure result
            return MultiSessionEvaluationResult(
                task_id=evaluation_task.task_id,
                task_completed=False,
                conversation=conversation,
                preference_usage={},
                stale_preference_usage=[],
                evaluation_event=evaluation_task.evaluation_event,
                rubric=evaluation_task.rubric,
                reasoning=f"Evaluation failed: {e}",
            )

    def _parse_judge_result(
        self,
        task_id: str,
        result: dict[str, Any],
        conversation: list[dict[str, str]],
        evaluation_event: "LifeEvent",
        rubric: "EvaluationRubric",
    ) -> MultiSessionEvaluationResult:
        """Parse the judge's JSON output into a result object."""
        # Extract preference usage
        current_usage = result.get("current_preference_usage", {})
        preference_usage = {
            pref_id: details.get("usage", "ignored")
            for pref_id, details in current_usage.items()
        }

        # Extract stale preference issues
        stale_usage = result.get("stale_preference_usage", {})
        stale_used = [
            pref_id
            for pref_id, details in stale_usage.items()
            if details.get("usage") == "stale_used"
        ]

        # Extract scores
        efficiency_score = float(result.get("efficiency_score", 0.0))
        preference_score = float(result.get("preference_score", 0.0))
        task_success = float(result.get("task_success_score", 0.0))
        stale_penalty = float(result.get("stale_penalty", 0.0))
        final_score = float(result.get("final_score", 0.0))

        # Extract turn classification counts
        productive_turns = int(result.get("productive_turns", 0))
        clarifying_turns = int(result.get("clarifying_turns", 0))

        return MultiSessionEvaluationResult(
            task_id=task_id,
            task_completed=result.get("task_completed", False),
            conversation=conversation,
            preference_usage=preference_usage,
            stale_preference_usage=stale_used,
            evaluation_event=evaluation_event,
            rubric=rubric,
            total_turns=int(result.get("total_turns", len(conversation))),
            productive_turns=productive_turns,
            clarifying_turns=clarifying_turns,
            correction_turns=int(result.get("correction_turns", 0)),
            efficiency_score=efficiency_score,
            preference_score=preference_score,
            stale_penalty=stale_penalty,
            task_success_score=task_success,
            final_score=final_score,
            reasoning=result.get("reasoning", ""),
        )


def evaluate_dialogue(
    evaluation_task: EvaluationTask,
    conversation: list[dict[str, str]],
    client: LLMClient | None = None,
) -> MultiSessionEvaluationResult:
    """Convenience function to evaluate a dialogue.

    Args:
        evaluation_task: The evaluation task with rubric
        conversation: The completed dialogue
        client: Optional LLM client

    Returns:
        Evaluation result with scores
    """
    judge = MultiSessionJudge(client)
    return judge.evaluate(evaluation_task, conversation)
