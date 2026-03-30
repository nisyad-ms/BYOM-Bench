"""
Multi-Session Judge for Evaluation.

The judge evaluates a completed dialogue using a Preference Judge LLM call
that validates simulator RECALLED/MISSED verdicts (specificity + stale checks).
"""

import json
import re
from typing import Any

from byom_bench.client import CONFIG, LLMClient, PooledLLMClient
from byom_bench.prompts import render_prompt
from byom_bench.schemas import (
    EvaluationTaskSpec,
    MultiSessionEvaluationResult,
)


def _extract_simulator_verdicts(
    conversation_with_scratchpads: list[dict[str, Any]],
    required_preferences: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract per-preference RECALLED/MISSED verdicts from simulator scratchpads.

    Supports both v10 (EVALUATION/TESTING/COVERED fields) and v11 (VERDICT field
    with current_pref_id on the turn dict) scratchpad formats. Auto-detects format
    by checking for the presence of ``current_pref_id`` on user turns.

    Returns a list of dicts: {pref_id, verdict}
    """
    pref_ids = {p["id"] for p in required_preferences}
    verdicts: dict[str, dict[str, str]] = {}

    # Detect format: v11 stores current_pref_id on user turn dicts
    is_v11 = any(
        turn.get("current_pref_id") is not None for turn in conversation_with_scratchpads if turn.get("role") == "user"
    )

    if is_v11:
        # --- v11 format: verdict + current_pref_id on turn ---
        for turn in conversation_with_scratchpads:
            if turn.get("role") != "user":
                continue
            scratchpad = turn.get("scratchpad")
            current_pref_id = turn.get("current_pref_id")
            if scratchpad is None:
                continue

            # Parse verdict from scratchpad
            if isinstance(scratchpad, dict):
                verdict_raw = scratchpad.get("verdict", "")
                proactive_raw = scratchpad.get("proactive_recall", [])
            else:
                verdict_raw = ""
                proactive_raw = []
                v_match = re.search(r"VERDICT:\s*(\S+)", str(scratchpad), re.IGNORECASE)
                if v_match:
                    verdict_raw = v_match.group(1).strip()
                p_match = re.search(r"PROACTIVE_RECALL:\s*(.+?)(?=\n|$)", str(scratchpad))
                if p_match:
                    value = p_match.group(1).strip()
                    if value.lower() != "none":
                        proactive_raw = re.findall(r"pref_\d+", value)

            # Record verdict for current_pref_id
            if current_pref_id and current_pref_id in pref_ids and current_pref_id not in verdicts:
                verdict_lower = str(verdict_raw).lower()
                if "recalled" in verdict_lower:
                    verdict = "recalled"
                elif "missed" in verdict_lower:
                    verdict = "missed"
                else:
                    # Skip N/A or unparseable (first turn)
                    verdict = None
                if verdict:
                    verdicts[current_pref_id] = {
                        "pref_id": current_pref_id,
                        "verdict": verdict,
                    }

            # Record proactive recalls
            proactive_ids = proactive_raw if isinstance(proactive_raw, list) else []
            for pid in proactive_ids:
                if pid in pref_ids and pid not in verdicts:
                    verdicts[pid] = {
                        "pref_id": pid,
                        "verdict": "recalled",
                    }
    else:
        # --- v10 format: EVALUATION/TESTING with off-by-one tracking ---
        prev_testing_pref_id: str | None = None

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

            # Use evaluation to resolve the PREVIOUS turn's testing pref
            if prev_testing_pref_id is not None and prev_testing_pref_id not in verdicts:
                if "N/A" not in evaluation:
                    eval_lower = evaluation.lower()
                    verdict = "recalled" if "recalled" in eval_lower else "missed"
                    verdicts[prev_testing_pref_id] = {
                        "pref_id": prev_testing_pref_id,
                        "verdict": verdict,
                    }

            # Track current testing pref for next iteration
            testing_pref_id: str | None = None
            for pid in pref_ids:
                if pid in testing:
                    testing_pref_id = pid
                    break
            prev_testing_pref_id = testing_pref_id

        # Also check for proactive recalls: preferences that moved to COVERED without being TESTING
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
                covered_list = (
                    [s.strip() for s in covered_match.group(1).split(",") if s.strip()] if covered_match else []
                )

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

    Uses a Preference Judge LLM call to validate simulator verdicts (specificity + stale checks).
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
        required_prefs_json = json.dumps(rubric.required_preferences, indent=2, ensure_ascii=False)
        transcript_json = json.dumps(conversation, indent=2, ensure_ascii=False)

        simulator_verdicts = _extract_simulator_verdicts(
            conversation_with_scratchpads or [], rubric.required_preferences
        )
        simulator_verdicts_json = json.dumps(simulator_verdicts, indent=2, ensure_ascii=False)

        try:
            pref_result = self._call_preference_judge(required_prefs_json, transcript_json, simulator_verdicts_json)

            return self._combine_results(
                evaluation_task.task_id,
                conversation,
                rubric.required_preferences,
                pref_result,
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
        simulator_verdicts_json: str,
    ) -> dict[str, Any]:
        """Call the preference judge LLM."""
        system_prompt = render_prompt("evaluation/preference_judge_system")
        user_prompt = render_prompt(
            "evaluation/preference_judge_user",
            required_preferences=required_prefs_json,
            transcript=transcript_json,
            simulator_verdicts=simulator_verdicts_json,
        )

        return self.client.complete_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=CONFIG["max_tokens"]["preference_judge"],
        )

    def _combine_results(
        self,
        task_id: str,
        conversation: list[dict[str, str]],
        required_preferences: list[dict[str, Any]],
        pref_result: dict[str, Any],
    ) -> MultiSessionEvaluationResult:
        """Combine results from preference judge and calculate scores."""
        preference_verdicts = pref_result.get("preference_verdicts", [])

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

        return MultiSessionEvaluationResult(
            task_id=task_id,
            conversation=conversation,
            preference_usage=preference_usage,
            stale_preference_usage=stale_used,
            preference_verdicts=preference_verdicts,
            stale_count=stale_count,
            recalled_count=recalled_count,
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
