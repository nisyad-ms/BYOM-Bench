"""Shared utilities for PersonaGym test scripts.

Includes:
- Logging configuration
- File naming and discovery for outputs
"""

import logging
import re
import sys
from pathlib import Path

SESSIONS_DIR = Path("outputs/conversation")
TASKS_DIR = Path("outputs/tasks")
EVAL_DIR = Path("outputs/evaluation")

SESSION_PATTERN = re.compile(r"^sessions_(\d{2})\.json$")
TASK_PATTERN = re.compile(r"^tasks_(\d{2})(?:_(\d{2}))?\.json$")
EVAL_PATTERN = re.compile(r"^eval_(\d{2})_(\d{2})_(\w+)\.json$")


def setup_logging(name: str) -> logging.Logger:
    """Configure console-only logging for imports."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))

    logger.addHandler(console_handler)

    return logger


def add_file_logging(logger: logging.Logger, log_dir: str = "logs") -> Path:
    """Add file handler to logger."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{logger.name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    logger.addHandler(file_handler)

    return log_file


def get_next_session_id() -> str:
    """Get the next available session ID (01, 02, etc.)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    existing = []
    for f in SESSIONS_DIR.iterdir():
        match = SESSION_PATTERN.match(f.name)
        if match:
            existing.append(int(match.group(1)))

    next_num = max(existing, default=0) + 1
    return f"{next_num:02d}"


def get_session_path(session_id: str | None = None) -> Path:
    """Get path for a session file."""
    sid = session_id or get_next_session_id()
    return SESSIONS_DIR / f"sessions_{sid}.json"


def get_task_path(session_id: str, task_num: int = 1) -> Path:
    """Get path for a task file (tasks_01_01.json, tasks_01_02.json, etc.)."""
    return TASKS_DIR / f"tasks_{session_id}_{task_num:02d}.json"


def get_eval_path(session_id: str, task_num: int, agent_type: str) -> Path:
    """Get path for an evaluation file (eval_01_01_context.json, etc.)."""
    return EVAL_DIR / f"eval_{session_id}_{task_num:02d}_{agent_type}.json"


def extract_session_id(filepath: Path | str) -> str | None:
    """Extract session ID from a session, task, or eval filename."""
    filename = Path(filepath).name

    match = SESSION_PATTERN.match(filename)
    if match:
        return match.group(1)

    match = TASK_PATTERN.match(filename)
    if match:
        return match.group(1)  # Returns session_id (e.g., "01")

    match = EVAL_PATTERN.match(filename)
    if match:
        return match.group(1)  # Returns session_id

    return None


def extract_task_num(filepath: Path | str) -> int | None:
    """Extract task number from a task or eval filename."""
    filename = Path(filepath).name

    match = TASK_PATTERN.match(filename)
    if match and match.group(2):
        return int(match.group(2))

    match = EVAL_PATTERN.match(filename)
    if match:
        return int(match.group(2))

    return None


def get_latest_session() -> Path | None:
    """Find the most recent session file (highest number)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for f in SESSIONS_DIR.iterdir():
        match = SESSION_PATTERN.match(f.name)
        if match:
            sessions.append((int(match.group(1)), f))

    if not sessions:
        return None

    sessions.sort(key=lambda x: x[0], reverse=True)
    return sessions[0][1]


def get_latest_task_for_session(session_id: str) -> Path | None:
    """Find the latest task file for a given session (highest task number)."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    matching = []
    for f in TASKS_DIR.iterdir():
        match = TASK_PATTERN.match(f.name)
        if match and match.group(1) == session_id:
            task_num = int(match.group(2)) if match.group(2) else 1
            matching.append((task_num, f))

    if not matching:
        return None

    matching.sort(key=lambda x: x[0], reverse=True)
    return matching[0][1]


def get_next_task_num(session_id: str) -> int:
    """Get the next available task number for a session."""
    latest = get_latest_task_for_session(session_id)
    if latest is None:
        return 1

    match = TASK_PATTERN.match(latest.name)
    if match and match.group(2):
        return int(match.group(2)) + 1
    return 2  # If old format without task_num, next is 2


def validate_task_session_match(session_path: Path, task_path: Path) -> bool:
    """Validate that a task file matches a session file."""
    session_id = extract_session_id(session_path)
    task_session_id = extract_session_id(task_path)

    if session_id is None or task_session_id is None:
        return False

    return session_id == task_session_id
