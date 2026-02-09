"""Shared utilities for PersonaGym test scripts.

Includes:
- Logging configuration
- File naming and discovery for outputs

Output structure:
    outputs/
        <date>_<HHMM>/           # e.g., 2026-02-02_1430
            sessions.json
            tasks/
                v1/
                    task_01.json
                    task_02.json
                v2/
                    task_01.json
            evaluations/
                <eval_timestamp>/
                    eval_01_context_01.json
                    eval_01_nocontext_01.json
                    run_config.json
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
LOGS_DIR = Path("logs")
SESSION_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")
TASK_PATTERN = re.compile(r"^task_(\d{2})\.json$")
EVAL_PATTERN = re.compile(r"^eval_(\d{2})_(\w+?)(?:_(\d{2}))?\.json$")
VERSION_PATTERN = re.compile(r"^v(\d+)$")
EVAL_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def setup_logging(name: str) -> logging.Logger:
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_latest_session_dir() -> Path | None:
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
    if session_name:
        path = OUTPUTS_DIR / session_name
        if path.is_dir():
            return path
        return None
    return get_latest_session_dir()


def get_session_path(session_dir: Path) -> Path:
    return session_dir / "sessions.json"


# --- Task versioning ---


def _get_task_versions(session_dir: Path) -> list[tuple[int, Path]]:
    tasks_dir = session_dir / "tasks"
    if not tasks_dir.exists():
        return []
    versions = []
    for d in tasks_dir.iterdir():
        if d.is_dir():
            match = VERSION_PATTERN.match(d.name)
            if match:
                versions.append((int(match.group(1)), d))
    versions.sort(key=lambda x: x[0])
    return versions


def get_latest_task_version(session_dir: Path) -> str | None:
    versions = _get_task_versions(session_dir)
    if not versions:
        return None
    return f"v{versions[-1][0]}"


def get_next_task_version(session_dir: Path) -> str:
    versions = _get_task_versions(session_dir)
    if not versions:
        return "v1"
    return f"v{versions[-1][0] + 1}"


def get_task_version_dir(session_dir: Path, version: str) -> Path:
    return session_dir / "tasks" / version


def get_task_path(session_dir: Path, task_num: int, version: str) -> Path:
    version_dir = get_task_version_dir(session_dir, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir / f"task_{task_num:02d}.json"


def get_all_tasks(session_dir: Path, version: str | None = None) -> list[Path]:
    if version is None:
        version = get_latest_task_version(session_dir)
    if version is None:
        return []

    version_dir = get_task_version_dir(session_dir, version)
    if not version_dir.exists():
        return []

    matching = []
    for f in version_dir.iterdir():
        match = TASK_PATTERN.match(f.name)
        if match:
            task_num = int(match.group(1))
            matching.append((task_num, f))

    matching.sort(key=lambda x: x[0])
    return [f for _, f in matching]


def get_next_task_num(session_dir: Path, version: str) -> int:
    tasks = get_all_tasks(session_dir, version)
    if not tasks:
        return 1

    last_task = tasks[-1]
    match = TASK_PATTERN.match(last_task.name)
    if match:
        return int(match.group(1)) + 1
    return 1


def get_tasks_by_nums(session_dir: Path, task_nums: str, version: str | None = None) -> list[Path]:
    if version is None:
        version = get_latest_task_version(session_dir)
    if version is None:
        return []
    paths = []
    for num_str in task_nums.split(","):
        num = int(num_str.strip())
        path = get_task_path(session_dir, num, version)
        if path.exists():
            paths.append(path)
    paths.sort(key=lambda p: p.name)
    return paths


def extract_task_num(filepath: Path | str) -> int | None:
    filename = Path(filepath).name

    match = TASK_PATTERN.match(filename)
    if match:
        return int(match.group(1))

    match = EVAL_PATTERN.match(filename)
    if match:
        return int(match.group(1))

    return None


# --- Evaluation run directories ---


def create_eval_run_dir(session_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    eval_run_dir = session_dir / "evaluations" / timestamp
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    return eval_run_dir


def _get_eval_run_dirs(session_dir: Path) -> list[tuple[str, Path]]:
    evals_dir = session_dir / "evaluations"
    if not evals_dir.exists():
        return []
    runs = []
    for d in evals_dir.iterdir():
        if d.is_dir() and EVAL_TIMESTAMP_PATTERN.match(d.name):
            runs.append((d.name, d))
    runs.sort(key=lambda x: x[0])
    return runs


def get_latest_eval_run_dir(session_dir: Path) -> Path | None:
    runs = _get_eval_run_dirs(session_dir)
    if not runs:
        return None
    return runs[-1][1]


def get_eval_run_dir(session_dir: Path, eval_run: str) -> Path | None:
    eval_run_dir = session_dir / "evaluations" / eval_run
    if eval_run_dir.is_dir():
        return eval_run_dir
    return None


def get_eval_path(eval_run_dir: Path, task_num: int, agent_type: str, run_id: int | None = None) -> Path:
    if run_id is not None:
        return eval_run_dir / f"eval_{task_num:02d}_{agent_type}_{run_id:02d}.json"
    return eval_run_dir / f"eval_{task_num:02d}_{agent_type}.json"


def save_eval_run_config(eval_run_dir: Path, config: dict) -> Path:
    config_path = eval_run_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path
