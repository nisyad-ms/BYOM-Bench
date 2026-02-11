"""
Multi-Session Judge for Evaluation.

The judge evaluates a completed dialogue using two separate LLM calls:
1. Preference Judge: Scores preference recall using First-Mention Rule
2. Efficiency Judge: Scores turn efficiency based on personalization vs generic responses
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import (
    EvaluationRubric,
    EvaluationTask,
    LifeEvent,
    MultiSessionEvaluationResult,
)


class MultiSessionJudge:
    """Evaluates agent performance on multi-session preference recall.

    Uses two separate LLM calls for cleaner separation of concerns:
    1. Preference Judge: First-mention analysis for preference recall
    2. Efficiency Judge: Turn classification for efficiency scoring
    """

    def __init__(self, client: LLMClient | PooledLLMClient | None = None):
        self.client = client or PooledLLMClient()

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
            with ThreadPoolExecutor(max_workers=2) as executor:
                pref_future = executor.submit(
                    self._call_preference_judge, required_prefs_json, transcript_json, num_required
                )
                eff_future = executor.submit(
                    self._call_efficiency_judge, required_prefs_json, transcript_json, agent_turns
                )
                pref_result = pref_future.result()
                eff_result = eff_future.result()

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
            print(f"Judge evaluation failed: {e}")
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
            "evaluation/preference_judge_user",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            num_required=num_required,
        )

        return self.client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=CONFIG["max_tokens"]["preference_judge"],
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
            "evaluation/efficiency_judge_user",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            agent_turns=agent_turns,
        )

        return self.client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=CONFIG["max_tokens"]["efficiency_judge"],
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
        """Combine results from preference and efficiency judges and calculate scores."""
        first_mention_trace = pref_result.get("first_mention_trace", [])
        turn_classifications = eff_result.get("turn_classifications", [])

        preference_usage = {entry.get("preference_id"): entry.get("usage", "ignored") for entry in first_mention_trace}

        stale_used = [entry.get("preference_id") for entry in first_mention_trace if entry.get("stale_used", False)]

        proactive_count, stale_count, preference_score = self._calculate_preference_score(first_mention_trace)

        turn_counts, efficiency_score = self._calculate_efficiency_score(turn_classifications, agent_turns)

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
            productive_turns=turn_counts["productive"],
            generic_turns=turn_counts["generic"],
            correction_turns=turn_counts["correction"],
            ignored_turns=turn_counts["ignored"],
            repeated_correction_turns=turn_counts["repeated_correction"],
            stale_count=stale_count,
            proactive_count=proactive_count,
            efficiency_score=efficiency_score,
            preference_score=preference_score,
        )

    def _calculate_preference_score(self, first_mention_trace: list[dict[str, Any]]) -> tuple[int, int, float]:
        """Calculate preference score from first_mention_trace.

        Returns:
            (proactive_count, stale_count, preference_score)
        """
        total = len(first_mention_trace)
        if total == 0:
            return 0, 0, 0.0

        proactive_count = sum(
            1 for e in first_mention_trace if e.get("usage") == "proactive" and not e.get("stale_used", False)
        )
        stale_count = sum(1 for e in first_mention_trace if e.get("stale_used", False))

        preference_score = max(0.0, (proactive_count - stale_count) / total)
        return proactive_count, stale_count, round(preference_score, 2)

    def _calculate_efficiency_score(
        self, turn_classifications: list[dict[str, Any]], agent_turns: int
    ) -> tuple[dict[str, int], float]:
        """Calculate efficiency score from turn_classifications.

        Returns:
            (turn_counts dict, efficiency_score)
        """
        counts = {
            "productive": 0,
            "generic": 0,
            "correction": 0,
            "ignored": 0,
            "repeated_correction": 0,
        }

        for tc in turn_classifications:
            turn_type = tc.get("type", "").lower()
            if turn_type == "productive":
                counts["productive"] += 1
            elif turn_type in ("generic", "clarifying_question"):
                counts["generic"] += 1
            elif turn_type == "correction":
                counts["correction"] += 1
            elif turn_type == "ignored":
                counts["ignored"] += 1
            elif turn_type == "repeated_correction":
                counts["repeated_correction"] += 1

        if counts["repeated_correction"] > 0:
            efficiency_score = 0.0
        elif agent_turns == 0:
            efficiency_score = 1.0
        else:
            penalty = 0.5 * counts["generic"] + 0.5 * counts["ignored"] + counts["correction"]
            efficiency_score = max(0.0, (agent_turns - penalty) / agent_turns)

        return counts, round(efficiency_score, 2)


def evaluate_dialogue(
    evaluation_task: EvaluationTask,
    conversation: list[dict[str, str]],
    client: LLMClient | PooledLLMClient | None = None,
) -> MultiSessionEvaluationResult:
    """Convenience function to evaluate a dialogue."""
    judge = MultiSessionJudge(client)
    return judge.evaluate(evaluation_task, conversation)
