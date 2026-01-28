"""Shared logging configuration for test scripts."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name: str) -> logging.Logger:
    """Configure console-only logging for imports.

    Args:
        name: Logger name (typically script name without .py)

    Returns:
        Configured logger with console output only
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))

    logger.addHandler(console_handler)

    return logger


def add_file_logging(logger: logging.Logger, log_dir: str = "logs") -> Path:
    """Add file handler to logger. Call this only when running as main script.

    Args:
        logger: Logger to add file handler to
        log_dir: Directory for log files

    Returns:
        Path to the log file
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{logger.name}_{timestamp}.log"

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    logger.addHandler(file_handler)

    return log_file

    return logger
