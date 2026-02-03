#!/usr/bin/env python
"""Render agent_system_with_context prompt using latest session data."""

import sys

sys.path.insert(0, ".")

from memory_gym.agents import ContextAwareAgent
from memory_gym.client import LLMClient
from memory_gym.schemas import MultiSessionOutput
from test_prompts._utils import load_latest_session, save_prompt


def main():
    raw, path = load_latest_session()
    print(f"Using session: {path}")

    data = MultiSessionOutput.from_dict(raw)
    agent = ContextAwareAgent(LLMClient())
    prompt = agent.build_context(data)

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    save_prompt(prompt, "agent_system_with_context")


if __name__ == "__main__":
    main()
