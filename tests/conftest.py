"""Shared fixtures and factory helpers for BYOM-Bench tests."""

import json
import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from byom_bench.schemas import (
    EvaluationRubric,
    EvaluationTaskSpec,
    ExpandedPersona,
    LifeEvent,
    MultiSessionEvaluationResult,
    MultiSessionOutput,
    Preference,
    PreferenceTimeline,
    Session,
)


# ---------------------------------------------------------------------------
# Factory helpers — plain functions, not fixtures
# ---------------------------------------------------------------------------

DOMAINS = ["work_habits", "health_body", "social_relationships", "leisure_hobbies", "daily_routines"]


def make_expanded_persona(**overrides) -> ExpandedPersona:
    defaults = dict(
        base_persona="A 30-year-old software engineer",
        age=30,
        gender="female",
        location="Portland, OR",
        work_education=["Works at a fintech startup", "Has a CS degree", "Remote worker"],
        health_wellness=["Runs 3x a week", "Mild pollen allergy", "Vegetarian"],
        family_relationships=["Lives with partner", "Has a dog", "Parents nearby"],
        baseline_preferences={
            d: [f"{d} preference {i}" for i in range(1, 6)] for d in DOMAINS
        },
    )
    defaults.update(overrides)
    return ExpandedPersona(**defaults)


def make_preference_timeline(n_baseline: int = 25, n_evolved: int = 3) -> PreferenceTimeline:
    """Build a timeline with baselines at session -1 and evolved chains."""
    timeline = PreferenceTimeline()
    for i in range(n_baseline):
        domain = DOMAINS[i % len(DOMAINS)]
        timeline.add_preference(
            fact=f"Baseline preference {i + 1}",
            domain=domain,
            session_id=-1,
            date="01/01/2026",
        )
    # Evolve the first n_evolved baselines at session 0
    for i in range(n_evolved):
        old_id = f"pref_{i + 1:03d}"
        timeline.evolve_preference(
            old_id=old_id,
            new_fact=f"Evolved preference replacing {old_id}",
            session_id=0,
            date="02/01/2026",
            reason=f"Life event changed preference {old_id}",
        )
    return timeline


def make_life_event(session_id: int = 0, **overrides) -> LifeEvent:
    defaults = dict(
        session_id=session_id,
        date="02/01/2026",
        event=f"Life event for session {session_id}",
        domain=DOMAINS[session_id % len(DOMAINS)],
    )
    defaults.update(overrides)
    return LifeEvent(**defaults)


def make_session(session_id: int, timeline: PreferenceTimeline, **overrides) -> Session:
    defaults = dict(
        session_id=session_id,
        life_event=make_life_event(session_id),
        conversation=[
            {"role": "user", "content": f"User message in session {session_id}"},
            {"role": "assistant", "content": f"Assistant reply in session {session_id}"},
            {"role": "user", "content": "Follow up question"},
            {"role": "assistant", "content": "Follow up answer"},
        ],
        active_preference_ids=timeline.get_preference_ids_at_session(session_id),
        new_preference_ids=[],
        evolved_preference_ids={},
        dropped_preference_ids=[],
    )
    defaults.update(overrides)
    return Session(**defaults)


def make_multisession_output(num_sessions: int = 2, n_baseline: int = 25, n_evolved: int = 3) -> MultiSessionOutput:
    timeline = make_preference_timeline(n_baseline=n_baseline, n_evolved=n_evolved)
    sessions = [make_session(i, timeline) for i in range(num_sessions)]
    life_events = [s.life_event for s in sessions]

    # Mark evolved prefs on session 0
    if n_evolved > 0 and num_sessions > 0:
        evolved_map = {}
        for i in range(n_evolved):
            old_id = f"pref_{i + 1:03d}"
            new_id = f"pref_{n_baseline + i + 1:03d}"
            evolved_map[old_id] = new_id
        sessions[0] = Session(
            session_id=sessions[0].session_id,
            life_event=sessions[0].life_event,
            conversation=sessions[0].conversation,
            active_preference_ids=sessions[0].active_preference_ids,
            new_preference_ids=[],
            evolved_preference_ids=evolved_map,
            dropped_preference_ids=[],
        )

    return MultiSessionOutput(
        persona="A 30-year-old software engineer",
        persona_id="test_persona_001",
        life_events=life_events,
        timeline=timeline,
        sessions=sessions,
        generation_timestamp="2026-01-01T00:00:00",
        expanded_persona=make_expanded_persona(),
    )


def make_evaluation_task_spec(mso: MultiSessionOutput | None = None) -> EvaluationTaskSpec:
    if mso is None:
        mso = make_multisession_output()

    required = []
    # Add evolved prefs
    for old_pref, new_pref in mso.get_evolved_preferences():
        required.append({
            "id": new_pref.preference_id,
            "fact": new_pref.fact,
            "supersedes": {"id": old_pref.preference_id, "fact": old_pref.fact},
        })
    # Add some baseline prefs
    active = mso.get_current_preferences()
    evolved_ids = mso.get_evolved_preference_ids()
    baselines = [p for p in active if p.preference_id not in evolved_ids]
    for p in baselines[:3]:
        required.append({"id": p.preference_id, "fact": p.fact})

    return EvaluationTaskSpec(
        task_id="eval_test1234",
        rubric=EvaluationRubric(required_preferences=required),
        persona=mso.persona,
    )


def make_evaluation_result(
    task_id: str = "eval_test1234",
    n_prefs: int = 6,
    n_recalled: int = 4,
    n_stale: int = 1,
) -> MultiSessionEvaluationResult:
    verdicts = []
    for i in range(n_prefs):
        verdict = "recalled" if i < n_recalled else "missed"
        verdicts.append({
            "preference_id": f"pref_{i + 1:03d}",
            "fact": f"Preference {i + 1}",
            "type": "evolved" if i < n_stale + 1 else "baseline",
            "final_verdict": verdict,
        })

    return MultiSessionEvaluationResult(
        task_id=task_id,
        conversation=[
            {"role": "user", "content": "Help me plan my day"},
            {"role": "assistant", "content": "I'd suggest starting with a morning run"},
        ],
        preference_usage={f"pref_{i + 1:03d}": ("recalled" if i < n_recalled else "missed") for i in range(n_prefs)},
        stale_preference_usage=[f"stale_pref_{i + 1:03d}" for i in range(n_stale)],
        preference_verdicts=verdicts,
        stale_count=n_stale,
        recalled_count=n_recalled,
        preference_score=max(0, (n_recalled - n_stale) / n_prefs) if n_prefs > 0 else 0.0,
        eval_seconds=5.2,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client():
    mock = MagicMock()
    mock.complete_chat.return_value = "I can help you with that."
    mock.complete_json.return_value = {"result": "ok"}
    return mock


@pytest.fixture
def mock_memory_store():
    mock = MagicMock()
    mock.populate.return_value = None
    mock.retrieve.return_value = ["Prefers morning runs", "Allergic to shellfish"]
    mock.cleanup.return_value = None
    return mock


@pytest.fixture
def sample_multisession_output():
    random.seed(42)
    return make_multisession_output()


@pytest.fixture
def sample_eval_task(sample_multisession_output):
    return make_evaluation_task_spec(sample_multisession_output)


@pytest.fixture
def tmp_outputs_dir(tmp_path):
    """Create a realistic outputs directory structure for testing."""
    import byom_bench.utils as utils

    old_outputs = utils.OUTPUTS_DIR
    utils.OUTPUTS_DIR = tmp_path

    session_name = "2026-01-01_120000_000001"
    session_dir = tmp_path / session_name
    session_dir.mkdir()

    # Write sessions.json
    mso = make_multisession_output()
    with open(session_dir / "sessions.json", "w") as f:
        json.dump(mso.to_dict(), f)

    # Write task
    task_dir = session_dir / "tasks" / "v1"
    task_dir.mkdir(parents=True)
    task_spec = make_evaluation_task_spec(mso)
    with open(task_dir / "task_01.json", "w") as f:
        json.dump(task_spec.to_dict(), f)

    # Write eval result
    eval_dir = session_dir / "evaluations" / "2026-01-01_130000"
    eval_dir.mkdir(parents=True)
    result = make_evaluation_result()
    with open(eval_dir / "eval_01_context_01.json", "w") as f:
        json.dump(result.to_dict(), f)

    # Write run_config
    with open(eval_dir / "run_config.json", "w") as f:
        json.dump({"agent": "context", "num_runs": 1}, f)

    yield tmp_path

    utils.OUTPUTS_DIR = old_outputs
