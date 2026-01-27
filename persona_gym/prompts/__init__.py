"""
Prompt management for PersonaGym.

This module provides a centralized system for loading and managing prompts
used across the PersonaGym pipeline. Prompts are stored as YAML files with
metadata describing their purpose and usage.

Usage:
    from persona_gym.prompts import load_prompt, render_prompt

    # Load a prompt template
    prompt = load_prompt("data_generation/personamemv2/generate_preferences")

    # Render with variables
    text = render_prompt(
        "data_generation/personamemv2/generate_preferences",
        persona="A software engineer...",
        topics="travel, coding",
        num_preferences=5,
    )
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Cache for loaded prompts
_prompt_cache: dict[str, dict[str, Any]] = {}

# Base directory for prompts
PROMPTS_DIR = Path(__file__).parent


def load_prompt(prompt_name: str, reload: bool = False) -> dict[str, Any]:
    """Load a prompt template from YAML file.

    Args:
        prompt_name: Path to prompt relative to prompts/ directory,
                    without .yaml extension (e.g., "data_generation/personamemv2/generate_preferences")
        reload: If True, bypass cache and reload from disk

    Returns:
        Dict containing prompt metadata and template

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not reload and prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]

    prompt_path = PROMPTS_DIR / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    _prompt_cache[prompt_name] = prompt_data
    logger.debug(f"Loaded prompt: {prompt_name}")

    return prompt_data


def render_prompt(prompt_name: str, **variables: Any) -> str:
    """Load and render a prompt template with variables.

    Args:
        prompt_name: Path to prompt relative to prompts/ directory
        **variables: Variables to substitute in the template

    Returns:
        Rendered prompt string

    Example:
        text = render_prompt(
            "data_generation/personamemv2/generate_preferences",
            persona="A software engineer...",
            topics="travel, coding",
        )
    """
    prompt_data = load_prompt(prompt_name)
    template = prompt_data.get("template", "")

    try:
        return template.format(**variables)
    except KeyError as e:
        logger.error(f"Missing variable in prompt {prompt_name}: {e}")
        raise ValueError(f"Missing required variable for prompt '{prompt_name}': {e}") from e


def get_prompt_metadata(prompt_name: str) -> dict[str, Any]:
    """Get metadata for a prompt without the template.

    Args:
        prompt_name: Path to prompt relative to prompts/ directory

    Returns:
        Dict with name, description, used_by, variables (excludes template)
    """
    prompt_data = load_prompt(prompt_name)
    return {k: v for k, v in prompt_data.items() if k != "template"}


def list_prompts(category: Optional[str] = None) -> list[str]:
    """List available prompts.

    Args:
        category: Optional category filter (e.g., "data_generation", "evaluation")

    Returns:
        List of prompt names (relative paths without .yaml)
    """
    search_dir = PROMPTS_DIR / category if category else PROMPTS_DIR
    prompts = []

    for yaml_file in search_dir.rglob("*.yaml"):
        relative_path = yaml_file.relative_to(PROMPTS_DIR)
        prompt_name = str(relative_path.with_suffix(""))
        prompts.append(prompt_name)

    return sorted(prompts)


def clear_cache() -> None:
    """Clear the prompt cache."""
    _prompt_cache.clear()
    logger.debug("Prompt cache cleared")
