"""Tests for PreferenceTimeline state machine."""

import pytest

from ream_bench.schemas import Preference, PreferenceTimeline


def test_add_preference_assigns_sequential_ids():
    tl = PreferenceTimeline()
    id1 = tl.add_preference(fact="Fact 1", domain="work_habits", session_id=0, date="01/01/2026")
    id2 = tl.add_preference(fact="Fact 2", domain="health_body", session_id=0, date="01/01/2026")
    id3 = tl.add_preference(fact="Fact 3", domain="leisure_hobbies", session_id=1, date="02/01/2026")

    assert id1 == "pref_001"
    assert id2 == "pref_002"
    assert id3 == "pref_003"


def test_add_preference_stores_all_fields():
    tl = PreferenceTimeline()
    pref_id = tl.add_preference(fact="Drinks green tea", domain="daily_routines", session_id=2, date="03/15/2026")

    pref = tl.preferences[pref_id]
    assert pref.fact == "Drinks green tea"
    assert pref.domain == "daily_routines"
    assert pref.created_at_session == 2
    assert pref.created_at_date == "03/15/2026"
    assert pref.is_active is True


def test_evolve_preference_creates_new_and_supersedes_old():
    tl = PreferenceTimeline()
    old_id = tl.add_preference(fact="Runs 3x a week", domain="health_body", session_id=-1, date="01/01/2026")
    new_id = tl.evolve_preference(
        old_id=old_id, new_fact="Runs 5x a week", session_id=2, date="03/01/2026", reason="Increased fitness"
    )

    old_pref = tl.preferences[old_id]
    new_pref = tl.preferences[new_id]

    assert old_pref.is_active is False
    assert old_pref.superseded_at_session == 2
    assert old_pref.superseded_by == new_id
    assert old_pref.reason_for_change == "Increased fitness"

    assert new_pref.is_active is True
    assert new_pref.fact == "Runs 5x a week"
    assert new_pref.domain == "health_body"  # inherited


def test_evolve_preference_with_new_domain():
    tl = PreferenceTimeline()
    old_id = tl.add_preference(fact="Works from office", domain="work_habits", session_id=-1, date="01/01/2026")
    new_id = tl.evolve_preference(
        old_id=old_id,
        new_fact="Works from home gym",
        session_id=1,
        date="02/01/2026",
        reason="Changed role",
        new_domain="health_body",
    )

    new_pref = tl.preferences[new_id]
    assert new_pref.domain == "health_body"


def test_evolve_preference_invalid_id_raises():
    tl = PreferenceTimeline()
    with pytest.raises(ValueError, match="pref_999"):
        tl.evolve_preference(old_id="pref_999", new_fact="New fact", session_id=0, date="01/01/2026")


def test_drop_preference():
    tl = PreferenceTimeline()
    pref_id = tl.add_preference(fact="Has a dog", domain="family_relationships", session_id=-1, date="01/01/2026")
    tl.drop_preference(pref_id, session_id=3, reason="Dog passed away")

    pref = tl.preferences[pref_id]
    assert pref.is_active is False
    assert pref.superseded_at_session == 3
    assert pref.superseded_by is None
    assert pref.reason_for_change == "Dog passed away"


def test_drop_preference_invalid_id_raises():
    tl = PreferenceTimeline()
    with pytest.raises(ValueError, match="pref_999"):
        tl.drop_preference("pref_999", session_id=0)


def test_get_active_preferences():
    tl = PreferenceTimeline()
    # Add 5 preferences
    ids = []
    for i in range(5):
        pid = tl.add_preference(fact=f"Pref {i}", domain="work_habits", session_id=-1, date="01/01/2026")
        ids.append(pid)

    # Evolve one (pref_001 -> pref_006)
    tl.evolve_preference(old_id=ids[0], new_fact="Evolved pref", session_id=1, date="02/01/2026")
    # Drop one (pref_002)
    tl.drop_preference(ids[1], session_id=2, reason="No longer relevant")

    active = tl.get_active_preferences()
    active_ids = {p.preference_id for p in active}

    # 5 original - 1 evolved - 1 dropped + 1 new from evolve = 4 active
    assert len(active) == 4
    assert ids[0] not in active_ids  # evolved away
    assert ids[1] not in active_ids  # dropped
    assert "pref_006" in active_ids  # the evolved replacement


def test_get_active_at_session():
    tl = PreferenceTimeline()
    # Add prefs at sessions 0, 1, 2
    id0 = tl.add_preference(fact="Session 0 pref", domain="work_habits", session_id=0, date="01/01/2026")
    id1 = tl.add_preference(fact="Session 1 pref", domain="health_body", session_id=1, date="02/01/2026")
    id2 = tl.add_preference(fact="Session 2 pref", domain="leisure_hobbies", session_id=2, date="03/01/2026")

    # Evolve session-0 pref at session 2
    new_id = tl.evolve_preference(
        old_id=id0, new_fact="Evolved session 0 pref", session_id=2, date="03/01/2026"
    )

    # At session 1: id0 still active (not yet evolved), id1 exists, id2 not yet created
    active_at_1 = tl.get_active_at_session(1)
    ids_at_1 = {p.preference_id for p in active_at_1}
    assert id0 in ids_at_1
    assert id1 in ids_at_1
    assert id2 not in ids_at_1
    assert new_id not in ids_at_1

    # At session 2: id0 superseded, id1 active, id2 active, new_id active
    active_at_2 = tl.get_active_at_session(2)
    ids_at_2 = {p.preference_id for p in active_at_2}
    assert id0 not in ids_at_2  # superseded at session 2, so not active at session 2
    assert id1 in ids_at_2
    assert id2 in ids_at_2
    assert new_id in ids_at_2


def test_get_preference_ids_at_session():
    tl = PreferenceTimeline()
    tl.add_preference(fact="Pref A", domain="work_habits", session_id=0, date="01/01/2026")
    tl.add_preference(fact="Pref B", domain="health_body", session_id=1, date="02/01/2026")

    ids_at_0 = tl.get_preference_ids_at_session(0)
    ids_at_1 = tl.get_preference_ids_at_session(1)

    assert ids_at_0 == ["pref_001"]
    assert set(ids_at_1) == {"pref_001", "pref_002"}
    assert all(isinstance(i, str) for i in ids_at_1)
