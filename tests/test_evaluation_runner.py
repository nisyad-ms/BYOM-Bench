"""Tests for evaluation runner helper functions (pure, no mocking needed)."""

from byom_bench.evaluation_multisession.runner import (
    _extract_proactive_recalls,
    _extract_verdict,
    _parse_plan_testing_order,
    _parse_scratchpad,
)


# ---------------------------------------------------------------------------
# _parse_plan_testing_order
# ---------------------------------------------------------------------------


class TestParsePlanTestingOrder:
    def test_parse_plan_testing_order_valid(self):
        text = "TESTING ORDER: [pref_001, pref_002, pref_003]"
        result = _parse_plan_testing_order(text, ["pref_001", "pref_002", "pref_003"])
        assert result == ["pref_001", "pref_002", "pref_003"]

    def test_parse_plan_testing_order_fallback(self):
        text = "No testing order here, just a plan."
        fallback = ["pref_010", "pref_020"]
        result = _parse_plan_testing_order(text, fallback)
        assert result == fallback

    def test_parse_plan_testing_order_missing_ids(self):
        """Parsed order is missing some IDs; they get appended at the end."""
        text = "TESTING ORDER: [pref_002, pref_003]"
        fallback = ["pref_001", "pref_002", "pref_003"]
        result = _parse_plan_testing_order(text, fallback)
        assert result == ["pref_002", "pref_003", "pref_001"]


# ---------------------------------------------------------------------------
# _extract_verdict
# ---------------------------------------------------------------------------


class TestExtractVerdict:
    def test_extract_verdict_recalled(self):
        assert _extract_verdict("VERDICT: RECALLED") == "recalled"

    def test_extract_verdict_missed(self):
        assert _extract_verdict("VERDICT: MISSED") == "missed"

    def test_extract_verdict_na(self):
        assert _extract_verdict("VERDICT: N/A") is None

    def test_extract_verdict_missing(self):
        assert _extract_verdict("No verdict field here") is None


# ---------------------------------------------------------------------------
# _extract_proactive_recalls
# ---------------------------------------------------------------------------


class TestExtractProactiveRecalls:
    def test_extract_proactive_recalls_valid(self):
        text = "PROACTIVE_RECALL: pref_005, pref_010, pref_099"
        uncovered = {"pref_005", "pref_010", "pref_020"}
        result = _extract_proactive_recalls(text, uncovered)
        assert result == ["pref_005", "pref_010"]

    def test_extract_proactive_recalls_none(self):
        text = "PROACTIVE_RECALL: none"
        result = _extract_proactive_recalls(text, {"pref_001"})
        assert result == []


# ---------------------------------------------------------------------------
# _parse_scratchpad
# ---------------------------------------------------------------------------


class TestParseScratchpad:
    def test_parse_scratchpad_v11(self):
        raw = (
            "VERDICT: RECALLED\n"
            "PROACTIVE_RECALL: pref_004, pref_007\n"
            "REASONING: The agent correctly referenced the user's morning routine."
        )
        result = _parse_scratchpad(raw)
        assert result["verdict"] == "RECALLED"
        assert result["proactive_recall"] == ["pref_004", "pref_007"]
        assert "morning routine" in result["reasoning"]

    def test_parse_scratchpad_v10(self):
        raw = (
            "COVERED: [pref_001, pref_002]\n"
            "UNCOVERED: [pref_003]\n"
            "EVALUATION: The agent missed the preference.\n"
            "ACTION: Ask about dietary restrictions.\n"
            "TESTING: pref_003 - Prefers gluten-free options"
        )
        result = _parse_scratchpad(raw)
        assert result["covered"] == ["pref_001", "pref_002"]
        assert result["uncovered"] == ["pref_003"]
        assert "missed" in result["evaluation"]
        assert "dietary" in result["action"]
        assert "pref_003" in result["testing"]

    def test_parse_scratchpad_raw_fallback(self):
        raw = "unparseable gibberish with no fields"
        result = _parse_scratchpad(raw)
        assert result == {"raw": raw}
