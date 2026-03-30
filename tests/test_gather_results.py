"""Tests for gather_results.py — stale rate computation and macro-averaging."""

import json
from pathlib import Path

import pytest

# Import the private function directly from the script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from gather_results import _compute_stale_rate


class TestComputeStaleRate:
    def test_basic(self):
        pref_scoring = {
            "stale_count": 2,
            "preference_verdicts": [
                {"type": "evolved"},
                {"type": "evolved"},
                {"type": "evolved"},
                {"type": "baseline"},
            ],
        }
        assert _compute_stale_rate(pref_scoring) == pytest.approx(2 / 3)

    def test_no_evolved_returns_none(self):
        pref_scoring = {
            "stale_count": 0,
            "preference_verdicts": [
                {"type": "baseline"},
                {"type": "baseline"},
            ],
        }
        assert _compute_stale_rate(pref_scoring) is None

    def test_zero_stale(self):
        pref_scoring = {
            "stale_count": 0,
            "preference_verdicts": [{"type": "evolved"}, {"type": "evolved"}],
        }
        assert _compute_stale_rate(pref_scoring) == 0.0

    def test_empty_verdicts(self):
        assert _compute_stale_rate({}) is None

    def test_uses_first_mention_trace_fallback(self):
        """Falls back to first_mention_trace when preference_verdicts missing."""
        pref_scoring = {
            "stale_count": 1,
            "first_mention_trace": [{"type": "evolved"}, {"type": "baseline"}],
        }
        assert _compute_stale_rate(pref_scoring) == 1.0


class TestGatherResultsIntegration:
    """Test the full gather pipeline using temporary eval files."""

    def _write_eval_file(self, eval_dir: Path, filename: str, pref_score: float, stale_count: int = 0, evolved_count: int = 2):
        verdicts = [{"type": "evolved"} for _ in range(evolved_count)] + [{"type": "baseline"} for _ in range(4)]
        data = {
            "task_id": "eval_test",
            "scores": {"preference_score": pref_score, "eval_seconds": 1.0},
            "conversation": [],
            "preference_scoring": {
                "recalled_count": int(pref_score * 6),
                "stale_count": stale_count,
                "stale_preference_usage": [],
                "preference_verdicts": verdicts,
            },
        }
        with open(eval_dir / filename, "w") as f:
            json.dump(data, f)

    def test_task_completion_binary(self, tmp_path):
        """task_completion is 1 only when preference_score is exactly 1.0."""
        session_dir = tmp_path / "2026-01-01_120000_000001"
        eval_dir = session_dir / "evaluations" / "2026-01-01_130000"
        eval_dir.mkdir(parents=True)

        self._write_eval_file(eval_dir, "eval_01_context_01.json", pref_score=1.0)
        self._write_eval_file(eval_dir, "eval_01_context_02.json", pref_score=0.99)
        self._write_eval_file(eval_dir, "eval_01_context_03.json", pref_score=0.0)

        from byom_bench.utils import EVAL_PATTERN

        results = []
        for f in sorted(eval_dir.glob("eval_*.json")):
            match = EVAL_PATTERN.match(f.name)
            if not match:
                continue
            with open(f) as fh:
                data = json.load(fh)
            score = data["scores"]["preference_score"]
            results.append(1 if score == 1.0 else 0)

        assert results == [1, 0, 0]

    def test_macro_averaging_runs_to_task(self):
        """3 runs with scores [0.5, 0.8, 1.0] average to ~0.767."""
        scores = [0.5, 0.8, 1.0]
        task_mean = sum(scores) / len(scores)
        assert task_mean == pytest.approx(0.7667, abs=0.001)

    def test_macro_averaging_tasks_to_session(self):
        """2 task means [0.5, 1.0] average to 0.75."""
        task_means = [0.5, 1.0]
        session_mean = sum(task_means) / len(task_means)
        assert session_mean == 0.75

    def test_macro_averaging_sessions_to_overall(self):
        """2 session means [0.6, 0.8] average to 0.7."""
        session_means = [0.6, 0.8]
        overall = sum(session_means) / len(session_means)
        assert overall == pytest.approx(0.7)
