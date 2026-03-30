"""Tests for byom_bench.utils — patterns, paths, and directory helpers."""

import byom_bench.utils as utils
from byom_bench.utils import (
    EVAL_PATTERN,
    SESSION_DIR_PATTERN,
    TASK_PATTERN,
    VERSION_PATTERN,
    create_session_dir,
    get_all_session_dirs,
    get_latest_eval_run_dir,
    get_latest_session_dir,
    get_next_task_version,
    get_session_dir,
)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


class TestSessionDirPattern:
    def test_session_dir_pattern_valid(self):
        assert SESSION_DIR_PATTERN.match("2026-02-02_143022")
        assert SESSION_DIR_PATTERN.match("2026-02-02_143022_123456")

    def test_session_dir_pattern_invalid(self):
        assert not SESSION_DIR_PATTERN.match("invalid")
        assert not SESSION_DIR_PATTERN.match("2026-02-02")
        assert not SESSION_DIR_PATTERN.match("abc_123456")


def test_task_pattern():
    m = TASK_PATTERN.match("task_01.json")
    assert m is not None
    assert m.group(1) == "01"


def test_eval_pattern():
    m = EVAL_PATTERN.match("eval_01_context_01.json")
    assert m is not None
    assert m.group(1) == "01"
    assert m.group(2) == "context"
    assert m.group(3) == "01"


def test_eval_pattern_no_run():
    m = EVAL_PATTERN.match("eval_01_context.json")
    assert m is not None
    assert m.group(1) == "01"
    assert m.group(2) == "context"
    assert m.group(3) is None


def test_version_pattern():
    assert VERSION_PATTERN.match("v1")
    assert VERSION_PATTERN.match("v12")
    assert not VERSION_PATTERN.match("version1")


# ---------------------------------------------------------------------------
# Directory helpers (use tmp_outputs_dir fixture from conftest)
# ---------------------------------------------------------------------------


def test_create_session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path)
    session_dir = create_session_dir()
    assert session_dir.exists()
    assert session_dir.parent == tmp_path
    assert SESSION_DIR_PATTERN.match(session_dir.name)


def test_get_session_dir_by_name(tmp_outputs_dir):
    result = get_session_dir("2026-01-01_120000_000001")
    assert result is not None
    assert result.name == "2026-01-01_120000_000001"


def test_get_all_session_dirs_sorted(tmp_outputs_dir):
    # Add a second session dir to verify sorting
    second = tmp_outputs_dir / "2026-01-02_100000"
    second.mkdir()
    dirs = get_all_session_dirs()
    names = [d.name for d in dirs]
    assert names == sorted(names)
    assert len(dirs) >= 2


def test_get_latest_session_dir(tmp_outputs_dir):
    # Add a newer session dir
    newer = tmp_outputs_dir / "2026-06-01_120000"
    newer.mkdir()
    result = get_latest_session_dir()
    assert result is not None
    assert result.name == "2026-06-01_120000"


def test_get_latest_session_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path)
    result = get_latest_session_dir()
    assert result is None


def test_get_next_task_version(tmp_outputs_dir):
    session_dir = tmp_outputs_dir / "2026-01-01_120000_000001"
    # v1 already exists from conftest fixture
    assert get_next_task_version(session_dir) == "v2"


def test_get_latest_eval_run_dir(tmp_outputs_dir):
    session_dir = tmp_outputs_dir / "2026-01-01_120000_000001"
    result = get_latest_eval_run_dir(session_dir)
    assert result is not None
    assert result.name == "2026-01-01_130000"
