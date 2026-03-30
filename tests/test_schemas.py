"""Tests for schema round-trips and serialization."""

from byom_bench.schemas import (
    EvaluationRubric,
    EvaluationTaskSpec,
    ExpandedPersona,
    LifeEvent,
    MultiSessionOutput,
    Preference,
)

from .conftest import (
    make_evaluation_result,
    make_evaluation_task_spec,
    make_expanded_persona,
    make_multisession_output,
)


def test_expanded_persona_round_trip():
    persona = make_expanded_persona()
    data = persona.to_dict()
    restored = ExpandedPersona.from_dict(data)

    assert restored.base_persona == persona.base_persona
    assert restored.age == persona.age
    assert restored.gender == persona.gender
    assert restored.location == persona.location
    assert restored.work_education == persona.work_education
    assert restored.health_wellness == persona.health_wellness
    assert restored.family_relationships == persona.family_relationships
    assert restored.baseline_preferences == persona.baseline_preferences


def test_expanded_persona_from_dict_defaults():
    persona = ExpandedPersona.from_dict({})

    assert persona.base_persona == ""
    assert persona.age == 0
    assert persona.gender == ""
    assert persona.location == ""
    assert persona.work_education == []
    assert persona.health_wellness == []
    assert persona.family_relationships == []
    assert persona.baseline_preferences == {}


def test_expanded_persona_to_full_description_with_prefs():
    persona = make_expanded_persona()
    description = persona.to_full_description(include_preferences=True)

    assert "Baseline Preferences:" in description
    for domain in persona.baseline_preferences:
        assert f"[{domain}]" in description


def test_expanded_persona_to_full_description_without_prefs():
    persona = make_expanded_persona()
    description = persona.to_full_description(include_preferences=False)

    assert "Baseline Preferences:" not in description
    assert "Work & Education:" in description
    assert "Health & Wellness:" in description
    assert "Family & Relationships:" in description


def test_life_event_round_trip():
    event = LifeEvent(
        session_id=3,
        date="05/15/2026",
        event="Started a new job",
        domain="work_habits",
        user_prompt="I just started a new job!",
    )
    data = event.to_dict()
    restored = LifeEvent.from_dict(data)

    assert restored.session_id == event.session_id
    assert restored.date == event.date
    assert restored.event == event.event
    assert restored.domain == event.domain
    assert restored.user_prompt == event.user_prompt


def test_life_event_user_prompt_conditional():
    event_no_prompt = LifeEvent(session_id=0, date="01/01/2026", event="Something happened")
    data_no_prompt = event_no_prompt.to_dict()
    assert "user_prompt" not in data_no_prompt

    event_with_prompt = LifeEvent(
        session_id=0, date="01/01/2026", event="Something happened", user_prompt="Tell me about it"
    )
    data_with_prompt = event_with_prompt.to_dict()
    assert "user_prompt" in data_with_prompt
    assert data_with_prompt["user_prompt"] == "Tell me about it"


def test_preference_is_active():
    active = Preference(
        preference_id="pref_001",
        fact="Likes coffee",
        domain="daily_routines",
        created_at_session=0,
        created_at_date="01/01/2026",
    )
    assert active.is_active is True

    superseded = Preference(
        preference_id="pref_002",
        fact="Likes tea",
        domain="daily_routines",
        created_at_session=0,
        created_at_date="01/01/2026",
        superseded_at_session=3,
        superseded_by="pref_010",
    )
    assert superseded.is_active is False


def test_evaluation_rubric_round_trip():
    required = [
        {"id": "pref_026", "fact": "Runs 5x a week", "supersedes": {"id": "pref_001", "fact": "Runs 3x a week"}},
        {"id": "pref_005", "fact": "Prefers dark mode"},
    ]
    rubric = EvaluationRubric(required_preferences=required)
    data = rubric.to_dict()
    restored = EvaluationRubric.from_dict(data)

    assert len(restored.required_preferences) == 2
    assert restored.required_preferences[0]["supersedes"]["id"] == "pref_001"
    assert restored.required_preferences[1]["fact"] == "Prefers dark mode"


def test_evaluation_task_spec_round_trip():
    task = make_evaluation_task_spec()
    data = task.to_dict()
    restored = EvaluationTaskSpec.from_dict(data)

    assert restored.task_id == task.task_id
    assert restored.persona == task.persona
    assert len(restored.rubric.required_preferences) == len(task.rubric.required_preferences)


def test_multisession_output_round_trip():
    mso = make_multisession_output(num_sessions=3, n_baseline=25, n_evolved=3)
    data = mso.to_dict()
    restored = MultiSessionOutput.from_dict(data)

    assert restored.persona == mso.persona
    assert restored.persona_id == mso.persona_id
    assert len(restored.sessions) == len(mso.sessions)
    assert restored.generation_timestamp == mso.generation_timestamp

    # Timeline preserves active and superseded counts
    assert len(restored.timeline.get_active_preferences()) == len(mso.timeline.get_active_preferences())
    superseded_orig = [p for p in mso.timeline.preferences.values() if not p.is_active]
    superseded_rest = [p for p in restored.timeline.preferences.values() if not p.is_active]
    assert len(superseded_rest) == len(superseded_orig)

    # Expanded persona preserved
    assert restored.expanded_persona is not None
    assert restored.expanded_persona.base_persona == mso.expanded_persona.base_persona


def test_multisession_output_round_trip_no_expanded_persona():
    mso = make_multisession_output()
    mso.expanded_persona = None
    data = mso.to_dict()
    restored = MultiSessionOutput.from_dict(data)

    assert restored.expanded_persona is None
    assert restored.persona == mso.persona


def test_evaluation_result_to_dict():
    result = make_evaluation_result(task_id="eval_abc123", n_prefs=6, n_recalled=4, n_stale=1)
    data = result.to_dict()

    assert data["task_id"] == "eval_abc123"
    assert "scores" in data
    assert data["scores"]["preference_score"] == result.preference_score
    assert "conversation" in data
    assert len(data["conversation"]) == 2
    assert "preference_scoring" in data
    assert data["preference_scoring"]["recalled_count"] == 4
    assert data["preference_scoring"]["stale_count"] == 1


def test_evaluation_result_error_field():
    result_ok = make_evaluation_result()
    data_ok = result_ok.to_dict()
    assert "error" not in data_ok

    result_err = make_evaluation_result()
    result_err.error = "LLM timeout"
    data_err = result_err.to_dict()
    assert data_err["error"] == "LLM timeout"
