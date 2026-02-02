"""
Multi-Session Judge for Evaluation.

The judge evaluates a completed dialogue using two separate LLM calls:
1. Preference Judge: Scores preference recall using First-Mention Rule
2. Efficiency Judge: Scores turn efficiency based on corrections and clarifying questions
"""

import json
import logging
from typing import Any

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

    Uses two separate LLM calls for cleaner separation of concerns:
    1. Preference Judge: First-mention analysis for preference recall
    2. Efficiency Judge: Turn classification for efficiency scoring
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

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
        agent_turns = sum(1 for turn in conversation if turn.get("role") == "assistant")
        required_prefs_json = json.dumps(rubric.required_preferences, indent=2, ensure_ascii=False)
        transcript_json = json.dumps(conversation, indent=2, ensure_ascii=False)
        num_required = len(rubric.required_preferences)

        try:
            pref_result = self._call_preference_judge(required_prefs_json, transcript_json, num_required)
            eff_result = self._call_efficiency_judge(required_prefs_json, transcript_json, agent_turns)

            return self._combine_results(
                evaluation_task.task_id,
                conversation,
                evaluation_task.evaluation_event,
                evaluation_task.rubric,
                pref_result,
                eff_result,
                agent_turns,
            )

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return MultiSessionEvaluationResult(
                task_id=evaluation_task.task_id,
                conversation=conversation,
                preference_usage={},
                stale_preference_usage=[],
                evaluation_event=evaluation_task.evaluation_event,
                rubric=evaluation_task.rubric,
                reasoning=f"Evaluation failed: {e}",
            )

    def _call_preference_judge(
        self,
        required_prefs_json: str,
        transcript_json: str,
        num_required: int,
    ) -> dict[str, Any]:
        """Call the preference judge LLM."""
        system_prompt = render_prompt("evaluation/preference_judge_system")
        user_prompt = render_prompt(
            "evaluation/preference_judge_instruction",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            num_required=num_required,
        )

        return self.client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
        )

    def _call_efficiency_judge(
        self,
        required_prefs_json: str,
        transcript_json: str,
        agent_turns: int,
    ) -> dict[str, Any]:
        """Call the efficiency judge LLM."""
        system_prompt = render_prompt("evaluation/efficiency_judge_system")
        user_prompt = render_prompt(
            "evaluation/efficiency_judge_instruction",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            agent_turns=agent_turns,
        )

        return self.client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
        )

    def _combine_results(
        self,
        task_id: str,
        conversation: list[dict[str, str]],
        evaluation_event: "LifeEvent",
        rubric: "EvaluationRubric",
        pref_result: dict[str, Any],
        eff_result: dict[str, Any],
        agent_turns: int,
    ) -> MultiSessionEvaluationResult:
        """Combine results from preference and efficiency judges."""
        pref_usage_raw = pref_result.get("preference_usage", {})
        preference_usage = {
            pref_id: details.get("usage", "ignored") if isinstance(details, dict) else details
            for pref_id, details in pref_usage_raw.items()
        }

        stale_used = [
            pref_id
            for pref_id, details in pref_usage_raw.items()
            if isinstance(details, dict) and details.get("stale_used", False)
        ]

        first_mention_trace = pref_result.get("first_mention_trace", [])
        turn_classifications = eff_result.get("turn_classifications", [])

        preference_score = float(pref_result.get("preference_score", 0.0))
        efficiency_score = float(eff_result.get("efficiency_score", 0.0))
        final_score = 0.5 * preference_score + 0.5 * efficiency_score

        pref_reasoning = pref_result.get("reasoning", "")
        eff_reasoning = eff_result.get("reasoning", "")
        combined_reasoning = f"Preference: {pref_reasoning}\nEfficiency: {eff_reasoning}"

        return MultiSessionEvaluationResult(
            task_id=task_id,
            conversation=conversation,
            preference_usage=preference_usage,
            stale_preference_usage=stale_used,
            evaluation_event=evaluation_event,
            rubric=rubric,
            first_mention_trace=first_mention_trace,
            turn_classifications=turn_classifications,
            total_turns=agent_turns,
            productive_turns=int(eff_result.get("productive_turns", 0)),
            clarifying_turns=int(eff_result.get("clarifying_turns", 0)),
            correction_turns=int(eff_result.get("correction_turns", 0)),
            ignored_turns=int(eff_result.get("ignored_turns", 0)),
            repeated_correction_turns=int(eff_result.get("repeated_correction_turns", 0)),
            stale_count=int(pref_result.get("stale_count", 0)),
            proactive_count=int(pref_result.get("proactive_count", 0)),
            efficiency_score=efficiency_score,
            preference_score=preference_score,
            final_score=final_score,
            reasoning=combined_reasoning,
        )


def evaluate_dialogue(
    evaluation_task: EvaluationTask,
    conversation: list[dict[str, str]],
    client: LLMClient | None = None,
) -> MultiSessionEvaluationResult:
    """Convenience function to evaluate a dialogue."""
    judge = MultiSessionJudge(client)
    return judge.evaluate(evaluation_task, conversation)
