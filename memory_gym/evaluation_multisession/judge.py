"""
Multi-Session Judge for Evaluation.

The judge evaluates a completed dialogue using two separate LLM calls:
1. Preference Judge: Validates simulator RECALLED/MISSED verdicts (specificity + stale checks)
2. Efficiency Judge: Scores turn efficiency based on personalization vs generic responses
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memory_gym.client import CONFIG, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import (
    EvaluationTaskSpec,
    MultiSessionEvaluationResult,
)


def _extract_simulator_verdicts(
    conversation_with_scratchpads: list[dict[str, Any]],
    required_preferences: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract per-preference RECALLED/MISSED verdicts from simulator scratchpads.

    Walks through user turns in the conversation with scratchpads. For each preference,
    finds the turn where it was TESTING and determines the verdict from the EVALUATION field.

    Returns a list of dicts: {pref_id, verdict, agent_quote, turn_number}
    """
    pref_ids = {p["id"] for p in required_preferences}
    verdicts: dict[str, dict[str, str]] = {}

    for turn in conversation_with_scratchpads:
        if turn.get("role") != "user":
            continue
        scratchpad = turn.get("scratchpad")
        if scratchpad is None:
            continue

        # Handle both raw string and parsed dict scratchpads
        if isinstance(scratchpad, dict):
            evaluation = scratchpad.get("evaluation", "")
            testing = scratchpad.get("testing", "")
            covered_list = scratchpad.get("covered", [])
        else:
            evaluation = ""
            testing = ""
            covered_list = []
            eval_match = re.search(r"EVALUATION:\s*(.+?)(?=\nACTION:|\nTESTING:|\Z)", str(scratchpad), re.DOTALL)
            if eval_match:
                evaluation = eval_match.group(1).strip()
            test_match = re.search(r"TESTING:\s*(.+)", str(scratchpad))
            if test_match:
                testing = test_match.group(1).strip()
            covered_match = re.search(r"COVERED:\s*\[([^\]]*)\]", str(scratchpad))
            if covered_match:
                covered_list = [s.strip() for s in covered_match.group(1).split(",") if s.strip()]

        # Skip initial "N/A" evaluation (turn 1)
        if "N/A" in evaluation:
            continue

        # Find which pref_id is being tested
        testing_pref_id = None
        for pid in pref_ids:
            if pid in testing:
                testing_pref_id = pid
                break

        if testing_pref_id is None or testing_pref_id in verdicts:
            continue

        # Determine verdict from evaluation text
        eval_lower = evaluation.lower()
        if "recalled" in eval_lower:
            verdict = "recalled"
        elif "missed" in eval_lower:
            verdict = "missed"
        else:
            verdict = "missed"

        verdicts[testing_pref_id] = {
            "pref_id": testing_pref_id,
            "verdict": verdict,
        }

    # Also check for proactive recalls: preferences that moved to COVERED without being TESTING
    # (agent proactively applied them). Walk through covered lists to find these.
    for turn in conversation_with_scratchpads:
        if turn.get("role") != "user":
            continue
        scratchpad = turn.get("scratchpad")
        if scratchpad is None:
            continue

        if isinstance(scratchpad, dict):
            covered_list = scratchpad.get("covered", [])
        else:
            covered_match = re.search(r"COVERED:\s*\[([^\]]*)\]", str(scratchpad))
            covered_list = [s.strip() for s in covered_match.group(1).split(",") if s.strip()] if covered_match else []

        for pid in covered_list:
            if pid in pref_ids and pid not in verdicts:
                verdicts[pid] = {
                    "pref_id": pid,
                    "verdict": "recalled",
                }

    # Fill in any preferences not found in scratchpads as missed
    result = []
    for p in required_preferences:
        pid = p["id"]
        if pid in verdicts:
            result.append(verdicts[pid])
        else:
            result.append(
                {
                    "pref_id": pid,
                    "verdict": "missed",
                }
            )

    return result


class MultiSessionJudge:
    """Evaluates agent performance on multi-session preference recall.

    Uses two separate LLM calls for cleaner separation of concerns:
    1. Preference Judge: Validates simulator verdicts (specificity + stale checks)
    2. Efficiency Judge: Turn classification for efficiency scoring
    """

    def __init__(self, client: LLMClient | PooledLLMClient | None = None):
        self.client = client or PooledLLMClient()

    def evaluate(
        self,
        evaluation_task: EvaluationTaskSpec,
        conversation: list[dict[str, str]],
        conversation_with_scratchpads: list[dict[str, Any]] | None = None,
    ) -> MultiSessionEvaluationResult:
        """Evaluate a completed dialogue.

        Args:
            evaluation_task: The evaluation task with rubric
            conversation: The dialogue transcript (clean, no scratchpads)
            conversation_with_scratchpads: Conversation with parsed scratchpad dicts on user turns.
                Used to extract simulator verdicts for the preference judge.

        Returns:
            MultiSessionEvaluationResult with scores and analysis
        """
        rubric = evaluation_task.rubric
        agent_turns = sum(1 for turn in conversation if turn.get("role") == "assistant")
        required_prefs_json = json.dumps(rubric.required_preferences, indent=2, ensure_ascii=False)
        transcript_json = json.dumps(conversation, indent=2, ensure_ascii=False)
        num_required = len(rubric.required_preferences)

        simulator_verdicts = _extract_simulator_verdicts(
            conversation_with_scratchpads or [], rubric.required_preferences
        )
        simulator_verdicts_json = json.dumps(simulator_verdicts, indent=2, ensure_ascii=False)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                pref_future = executor.submit(
                    self._call_preference_judge,
                    required_prefs_json,
                    transcript_json,
                    num_required,
                    simulator_verdicts_json,
                )
                eff_future = executor.submit(
                    self._call_efficiency_judge, required_prefs_json, transcript_json, agent_turns
                )
                pref_result = pref_future.result()
                eff_result = eff_future.result()

            return self._combine_results(
                evaluation_task.task_id,
                conversation,
                rubric.required_preferences,
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
                reasoning=f"Evaluation failed: {e}",
                error=str(e),
            )

    def _call_preference_judge(
        self,
        required_prefs_json: str,
        transcript_json: str,
        num_required: int,
        simulator_verdicts_json: str,
    ) -> dict[str, Any]:
        """Call the preference judge LLM."""
        system_prompt = render_prompt("evaluation/preference_judge_system")
        user_prompt = render_prompt(
            "evaluation/preference_judge_user",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            num_required=num_required,
            simulator_verdicts=simulator_verdicts_json,
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
        required_preferences: list[dict[str, Any]],
        pref_result: dict[str, Any],
        eff_result: dict[str, Any],
        agent_turns: int,
    ) -> MultiSessionEvaluationResult:
        """Combine results from preference and efficiency judges and calculate scores."""
        preference_verdicts = pref_result.get("preference_verdicts", [])
        turn_classifications = eff_result.get("turn_classifications", [])

        # Enrich trace entries with preference content and type for easier manual review
        pref_lookup = {p["id"]: p for p in required_preferences}
        for entry in preference_verdicts:
            pref_id = entry.get("preference_id")
            if pref_id and pref_id in pref_lookup:
                pref = pref_lookup[pref_id]
                entry["preference"] = pref["fact"]
                if "supersedes" in pref:
                    entry["type"] = "evolved"
                    entry["supersedes"] = pref["supersedes"]
                else:
                    entry["type"] = "baseline"

        preference_usage = {
            entry["preference_id"]: entry.get("final_verdict", "missed")
            for entry in preference_verdicts
            if entry.get("preference_id")
        }

        stale_used = [
            entry["preference_id"]
            for entry in preference_verdicts
            if entry.get("stale_used", False) and entry.get("preference_id")
        ]

        recalled_count, stale_count, preference_score = self._calculate_preference_score(preference_verdicts)

        turn_counts, efficiency_score = self._calculate_efficiency_score(turn_classifications, agent_turns)

        return MultiSessionEvaluationResult(
            task_id=task_id,
            conversation=conversation,
            preference_usage=preference_usage,
            stale_preference_usage=stale_used,
            preference_verdicts=preference_verdicts,
            turn_classifications=turn_classifications,
            total_turns=agent_turns,
            productive_turns=turn_counts["productive"],
            generic_turns=turn_counts["generic"],
            correction_turns=turn_counts["correction"],
            ignored_turns=turn_counts["ignored"],
            repeated_correction_turns=turn_counts["repeated_correction"],
            stale_count=stale_count,
            recalled_count=recalled_count,
            efficiency_score=efficiency_score,
            preference_score=preference_score,
        )

    def _calculate_preference_score(self, preference_verdicts: list[dict[str, Any]]) -> tuple[int, int, float]:
        """Calculate preference score from preference_verdicts.

        Returns:
            (recalled_count, stale_count, preference_score)
        """
        total = len(preference_verdicts)
        if total == 0:
            return 0, 0, 0.0

        recalled_count = sum(1 for e in preference_verdicts if e.get("final_verdict") == "recalled")
        stale_count = sum(1 for e in preference_verdicts if e.get("stale_used", False))

        preference_score = max(0.0, (recalled_count - stale_count) / total)
        return recalled_count, stale_count, round(preference_score, 2)

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
