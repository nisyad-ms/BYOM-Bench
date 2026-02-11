"""
Prompt management for PersonaGym.

This module provides a centralized system for loading and managing prompts
used across the PersonaGym pipeline. Prompts are stored as YAML files with
metadata describing their purpose and usage.

Prompt versions are controlled via prompt_config.yaml at project root.

Usage:
    from memory_gym.prompts import render_prompt

    # Render with variables (version resolved from config)
    text = render_prompt(
        "data_generation/multisession/expand_persona_instruction",
        persona="A software engineer...",
    )
"""

from pathlib import Path
from typing import Any

import yaml

_prompt_cache: dict[str, dict[str, Any]] = {}
_config_cache: dict[str, str] | None = None

PROMPTS_DIR = Path(__file__).parent
PROJECT_ROOT = PROMPTS_DIR.parent.parent


def _load_prompt_config() -> dict[str, str]:
    """Load prompt version configuration from prompt_config.yaml."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = PROJECT_ROOT / "configs" / "prompt_config.yaml"
    if not config_path.exists():
        _config_cache = {}
        return _config_cache

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _config_cache = config.get("prompts", {})
    return _config_cache


def _resolve_prompt_name(prompt_name: str) -> str:
    """Resolve prompt name to versioned name based on config.

    Args:
        prompt_name: Base prompt name (e.g., "data_generation/multisession/generate_life_story_instruction")

    Returns:
        Versioned prompt name (e.g., "data_generation/multisession/generate_life_story_instruction_v2")
    """
    config = _load_prompt_config()
    version = config.get(prompt_name, "")

    if version:
        return f"{prompt_name}_{version}"

    return prompt_name


def reload_config() -> None:
    """Force reload of prompt configuration. Useful for testing."""
    global _config_cache
    _config_cache = None
    _load_prompt_config()


def load_prompt(prompt_name: str, reload: bool = False, use_config: bool = True) -> dict[str, Any]:
    """Load a prompt template from YAML file.

    Args:
        prompt_name: Path to prompt relative to prompts/ directory,
                    without .yaml extension (e.g., "data_generation/multisession/expand_persona_instruction")
        reload: If True, bypass cache and reload from disk
        use_config: If True, resolve version from prompt_config.yaml

    Returns:
        Dict containing prompt metadata and template

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if use_config:
        resolved_name = _resolve_prompt_name(prompt_name)
    else:
        resolved_name = prompt_name

    if not reload and resolved_name in _prompt_cache:
        return _prompt_cache[resolved_name]

    prompt_path = PROMPTS_DIR / f"{resolved_name}.yaml"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    _prompt_cache[resolved_name] = prompt_data

    return prompt_data


def render_prompt(prompt_name: str, use_config: bool = True, **variables: Any) -> str:
    """Load and render a prompt template with variables.

    Args:
        prompt_name: Path to prompt relative to prompts/ directory
        use_config: If True, resolve version from prompt_config.yaml
        **variables: Variables to substitute in the template

    Returns:
        Rendered prompt string

    Example:
        text = render_prompt(
            "data_generation/multisession/expand_persona_instruction",
            persona="A software engineer...",
        )
    """
    prompt_data = load_prompt(prompt_name, use_config=use_config)
    template = prompt_data.get("template", "")

    try:
        return template.format(**variables)
    except KeyError as e:
        raise ValueError(f"Missing required variable for prompt '{prompt_name}': {e}") from e
