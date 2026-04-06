"""Store registry — autodiscovers MemoryStore implementations.

Scans this package for classes ending in ``MemoryStore`` that have a
``_sentinel_agent_type`` class variable (set by ``SentinelMixin``).
Modules whose imports fail (missing optional dependencies) are silently
skipped.
"""

import importlib
import pkgutil
from pathlib import Path

from .foundry import FoundryMemoryStore, get_foundry_configs
from .protocol import MemoryStore

_STORE_REGISTRY: dict[str, type] = {}


def _discover_stores() -> None:
    """Scan stores/ for MemoryStore classes with ``_sentinel_agent_type``."""
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_") or info.name == "protocol":
            continue
        try:
            mod = importlib.import_module(f".{info.name}", __package__)
        except ImportError:
            continue  # optional dependency not installed
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("MemoryStore")
                and attr_name != "MemoryStore"
                and hasattr(obj, "_sentinel_agent_type")
            ):
                _STORE_REGISTRY[obj._sentinel_agent_type] = obj


_discover_stores()


def get_store_class(agent_type: str) -> type | None:
    """Return the store class registered under *agent_type*, or None."""
    return _STORE_REGISTRY.get(agent_type)


def get_available_agent_types() -> list[str]:
    """Return sorted list of autodiscovered agent type names."""
    return sorted(_STORE_REGISTRY.keys())


__all__ = [
    "FoundryMemoryStore",
    "MemoryStore",
    "get_available_agent_types",
    "get_foundry_configs",
    "get_store_class",
]
