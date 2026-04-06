"""Shared sentinel file helpers for memory stores.

All MemoryStore implementations track ingestion progress via a JSON sentinel
file so that resumed evaluation runs can skip re-population. This mixin
provides the four common helpers: path resolution, read, write, delete.

Usage::

    class MyMemoryStore(SentinelMixin):
        _sentinel_agent_type = "my_store"

        def __init__(self, ..., sentinel_dir, session_name):
            self._sentinel_dir = sentinel_dir
            self._session_name = session_name
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SentinelMixin:
    """Mixin providing sentinel file management for MemoryStore classes.

    Subclasses must:
    - Set ``_sentinel_agent_type`` as a class variable (e.g. ``"mem0"``).
    - Set ``self._sentinel_dir`` and ``self._session_name`` in ``__init__``.
    """

    _sentinel_agent_type: str
    _sentinel_dir: Path | None
    _session_name: str | None

    @property
    def _sentinel_path(self) -> Path | None:
        if self._sentinel_dir is None or self._session_name is None:
            return None
        return self._sentinel_dir / f"{self._session_name}_{self._sentinel_agent_type}.json"

    def _read_sentinel(self) -> dict[str, Any] | None:
        path = self._sentinel_path
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_sentinel(self, sessions_ingested: int, **extra: Any) -> None:
        path = self._sentinel_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_type": self._sentinel_agent_type,
            "sessions_ingested": sessions_ingested,
            **extra,
            "timestamp": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    def _delete_sentinel(self) -> None:
        path = self._sentinel_path
        if path is not None and path.exists():
            path.unlink()
