"""Shared utilities for PersonaGym test scripts.

Includes:
- Logging configuration
- File naming and discovery for outputs

Output structure:
    outputs/
        prompts/
        <date>_<HHMM>/           # e.g., 2026-02-02_1430
            sessions.json
            tasks/
                task_01.json
                task_02.json
            evaluation/
                eval_01_context.json
                eval_01_nocontext.json
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
LOGS_DIR = Path("logs")
SESSION_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")
TASK_PATTERN = re.compile(r"^task_(\d{2})\.json$")
EVAL_PATTERN = re.compile(r"^eval_(\d{2})_(\w+)\.json$")


def setup_logging(name: str) -> logging.Logger:
    """Configure console-only logging for imports."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(console_handler)

    persona_gym_logger = logging.getLogger("persona_gym")
    if not persona_gym_logger.handlers:
        persona_gym_logger.setLevel(logging.INFO)
        persona_gym_logger.addHandler(console_handler)

    return logger


def add_file_logging(logger: logging.Logger, session_dir: Path | None = None) -> Path:
    """Add file handler to logger and persona_gym logger.

    Args:
        logger: The logger to add file handler to
        session_dir: Session directory to mirror in logs/. If None, uses flat logs/ dir.
    """
    if session_dir:
        log_path = LOGS_DIR / session_dir.name
    else:
        log_path = LOGS_DIR
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{logger.name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)

    persona_gym_logger = logging.getLogger("persona_gym")
    persona_gym_logger.addHandler(file_handler)

    return log_file


def create_session_dir() -> Path:
    """Create a new session directory with timestamp (e.g., 2026-02-02_1430)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_latest_session_dir() -> Path | None:
    """Find the most recent session directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dirs = []
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir() and SESSION_DIR_PATTERN.match(d.name):
            dirs.append(d)

    if not dirs:
        return None

    dirs.sort(key=lambda x: x.name, reverse=True)
    return dirs[0]


def get_all_session_dirs() -> list[Path]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and SESSION_DIR_PATTERN.match(d.name)]
    dirs.sort(key=lambda x: x.name)
    return dirs


def get_session_dir(session_name: str | None) -> Path | None:
    """Get session directory from session name or find latest.

    Args:
        session_name: Session folder name (e.g., '2026-02-02_1414') or None for latest

    Returns:
        Path to session directory or None if not found
    """
    if session_name:
        path = OUTPUTS_DIR / session_name
        if path.is_dir():
            return path
        return None
    return get_latest_session_dir()


def get_session_path(session_dir: Path) -> Path:
    """Get path for sessions.json in a session directory."""
    return session_dir / "sessions.json"


def get_task_path(session_dir: Path, task_num: int) -> Path:
    """Get path for a task file."""
    tasks_dir = session_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir / f"task_{task_num:02d}.json"


def get_eval_path(session_dir: Path, task_num: int, agent_type: str, run_id: int | None = None) -> Path:
    """Get path for an evaluation file.

    Args:
        session_dir: Session directory
        task_num: Task number
        agent_type: Agent type (context, nocontext, foundry)
        run_id: Optional run ID for multiple runs (1, 2, 3, etc.)

    Returns:
        Path like eval_04_context.json or eval_04_context_01.json (with run_id)
    """
    eval_dir = session_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        return eval_dir / f"eval_{task_num:02d}_{agent_type}_{run_id:02d}.json"
    return eval_dir / f"eval_{task_num:02d}_{agent_type}.json"


def extract_task_num(filepath: Path | str) -> int | None:
    """Extract task number from a task or eval filename."""
    filename = Path(filepath).name

    match = TASK_PATTERN.match(filename)
    if match:
        return int(match.group(1))

    match = EVAL_PATTERN.match(filename)
    if match:
        return int(match.group(1))

    return None


def get_all_tasks(session_dir: Path) -> list[Path]:
    """Get all task files in a session directory, sorted by task number."""
    tasks_dir = session_dir / "tasks"
    if not tasks_dir.exists():
        return []

    matching = []
    for f in tasks_dir.iterdir():
        match = TASK_PATTERN.match(f.name)
        if match:
            task_num = int(match.group(1))
            matching.append((task_num, f))

    matching.sort(key=lambda x: x[0])
    return [f for _, f in matching]


def get_next_task_num(session_dir: Path) -> int:
    """Get the next available task number for a session."""
    tasks = get_all_tasks(session_dir)
    if not tasks:
        return 1

    last_task = tasks[-1]
    match = TASK_PATTERN.match(last_task.name)
    if match:
        return int(match.group(1)) + 1
    return 1


def get_tasks_by_nums(session_dir: Path, task_nums: str) -> list[Path]:
    paths = []
    for num_str in task_nums.split(","):
        num = int(num_str.strip())
        path = get_task_path(session_dir, num)
        if path.exists():
            paths.append(path)
    paths.sort(key=lambda p: p.name)
    return paths
