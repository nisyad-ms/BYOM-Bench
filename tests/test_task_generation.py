"""Tests for EvaluationTaskGenerator."""

import random
import re

import pytest

from byom_bench.schemas import EvaluationTaskSpec
from byom_bench.task_generators.evaluation_task import EvaluationTaskGenerator
from tests.conftest import make_multisession_output

TASK_ID_RE = re.compile(r"^eval_[0-9a-f]{8}$")


@pytest.fixture
def generator():
    return EvaluationTaskGenerator()


# --- batch generation ---


def test_generate_batch_returns_correct_count(generator):
    random.seed(42)
    data = make_multisession_output()
    tasks = generator.generate_batch(data, num_tasks=5)
    assert len(tasks) == 5
    assert all(isinstance(t, EvaluationTaskSpec) for t in tasks)


def test_task_ids_are_unique(generator):
    random.seed(42)
    data = make_multisession_output()
    tasks = generator.generate_batch(data, num_tasks=5)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_task_ids_format(generator):
    random.seed(42)
    data = make_multisession_output()
    tasks = generator.generate_batch(data, num_tasks=5)
    for t in tasks:
        assert TASK_ID_RE.match(t.task_id), f"task_id {t.task_id!r} does not match eval_[0-9a-f]{{8}}"


def test_rubric_pref_count(generator):
    random.seed(42)
    data = make_multisession_output()
    tasks = generator.generate_batch(data, num_tasks=5, prefs_per_task=6)
    for t in tasks:
        assert len(t.rubric.required_preferences) == 6


def test_evolved_prefs_have_supersedes(generator):
    random.seed(42)
    data = make_multisession_output()  # 25 baseline, 3 evolved
    tasks = generator.generate_batch(data, num_tasks=5, prefs_per_task=6)

    evolved_ids = data.get_evolved_preference_ids()
    for t in tasks:
        for pref in t.rubric.required_preferences:
            if pref["id"] in evolved_ids:
                assert "supersedes" in pref, f"Evolved pref {pref['id']} missing supersedes"


def test_baseline_prefs_no_supersedes(generator):
    random.seed(42)
    data = make_multisession_output()
    tasks = generator.generate_batch(data, num_tasks=5, prefs_per_task=6)

    evolved_ids = data.get_evolved_preference_ids()
    for t in tasks:
        for pref in t.rubric.required_preferences:
            if pref["id"] not in evolved_ids:
                assert "supersedes" not in pref, f"Baseline pref {pref['id']} should not have supersedes"


# --- single generation ---


def test_generate_single_task(generator):
    random.seed(42)
    data = make_multisession_output()
    task = generator.generate(data)
    assert isinstance(task, EvaluationTaskSpec)
    assert TASK_ID_RE.match(task.task_id)


# --- edge cases ---


def test_handles_no_evolved_prefs(generator):
    random.seed(42)
    data = make_multisession_output(n_evolved=0)
    tasks = generator.generate_batch(data, num_tasks=3)
    assert len(tasks) == 3
    for t in tasks:
        for pref in t.rubric.required_preferences:
            assert "supersedes" not in pref


def test_handles_few_prefs(generator):
    random.seed(42)
    data = make_multisession_output(n_baseline=3, n_evolved=0)
    tasks = generator.generate_batch(data, num_tasks=2, prefs_per_task=6)
    assert len(tasks) == 2
    for t in tasks:
        # Should gracefully reduce to available prefs (3 baselines, 0 evolved)
        assert len(t.rubric.required_preferences) <= 6
        assert len(t.rubric.required_preferences) > 0
