#!/usr/bin/env python
"""Test data generation - Generate multi-session conversation data with preference evolution.

Usage:
    python test_data_generation.py                                    # 1 random persona
    python test_data_generation.py --persona all                      # All from base_personas.json
    python test_data_generation.py --persona all --num 10             # First 10 across all domains
    python test_data_generation.py --persona "software development"   # All from that domain
    python test_data_generation.py --persona "software development" --num 5  # First 5 from domain
"""

import argparse
import json
import random
import time
from pathlib import Path

from memory_gym.utils import create_session_dir, get_session_path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_PERSONAS_FILE = DATA_DIR / "base_personas.json"


def _load_base_personas() -> dict[str, list[str]]:
    if not BASE_PERSONAS_FILE.exists():
        raise FileNotFoundError(f"Base personas file not found: {BASE_PERSONAS_FILE}")
    with open(BASE_PERSONAS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _flatten(data: dict[str, list[str]]) -> list[str]:
    return [p for personas in data.values() for p in personas]


def resolve_personas(persona_arg: str | None) -> list[str]:
    """Resolve --persona argument into a list of persona strings.

    Returns a list with one or more personas. If persona_arg is None,
    returns a single random persona from base_personas.json.
    """
    data = _load_base_personas()

    if persona_arg is None:
        all_personas = _flatten(data)
        if not all_personas:
            raise ValueError("No personas found in base_personas.json")
        return [random.choice(all_personas)]

    if persona_arg == "all":
        all_personas = _flatten(data)
        if not all_personas:
            raise ValueError("No personas found in base_personas.json")
        return all_personas

    if persona_arg in data:
        personas = data[persona_arg]
        if not personas:
            raise ValueError(f"No personas found for domain '{persona_arg}'")
        return personas

    raise ValueError(
        f"Unknown persona value: '{persona_arg}'. Use 'all' or a domain name: {list(data.keys())}"
    )


def run_one(persona: str, num_sessions: int) -> Path:
    """Generate multi-session data for a single persona. Returns session dir."""
    from memory_gym.data_generators import MultiSessionGenerator

    session_dir = create_session_dir()
    output_path = get_session_path(session_dir)

    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=num_sessions,
    )

    result = generator.generate_multi_session()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return session_dir


def main():
    parser = argparse.ArgumentParser(description="Test multi-session data generation")
    parser.add_argument("--sessions", type=int, default=2, help="Number of sessions per persona")
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="'all' or a domain name. Default: 1 random persona.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Use first N personas from the resolved list. Only valid with all/domain.",
    )
    args = parser.parse_args()

    personas = resolve_personas(args.persona)

    # --num only makes sense for list modes
    if args.num is not None:
        if args.persona is None:
            parser.error("--num requires --persona (test, all, or a domain name)")
        personas = personas[: args.num]

    print(f"Generating data for {len(personas)} persona(s), {args.sessions} sessions each\n")

    for i, persona in enumerate(personas, 1):
        print(f"[{i}/{len(personas)}] {persona}")
        session_dir = run_one(persona, args.sessions)
        print(f"  -> {session_dir}\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
