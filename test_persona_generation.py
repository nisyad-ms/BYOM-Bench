#!/usr/bin/env python
"""Test persona generation only — expand a base persona into life facts + baseline preferences.

Usage:
    python test_persona_generation.py                                    # 1 random persona
    python test_persona_generation.py --persona test                     # All from test set
    python test_persona_generation.py --persona test --num 2             # First 2 from test set
    python test_persona_generation.py --persona all --num 10             # First 10 across all domains
    python test_persona_generation.py --persona "software development"   # All from that domain
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from test_data_generation import resolve_personas

from memory_gym.data_generators import MultiSessionGenerator

OUTPUTS_DIR = Path("outputs")


def run_one(persona: str) -> tuple[str, Path]:
    """Expand a single persona and save to outputs/. Returns (persona, output_path)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    output_path = session_dir / "expanded_persona.json"

    generator = MultiSessionGenerator(persona=persona)
    expanded = generator._expand_persona()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(expanded.to_dict(), f, indent=2, ensure_ascii=False)

    return persona, output_path


def main():
    parser = argparse.ArgumentParser(description="Test persona expansion (life facts + baseline preferences)")
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="'test', 'all', or a domain name. Default: 1 random persona.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Use first N personas from the resolved list.",
    )
    args = parser.parse_args()

    personas = resolve_personas(args.persona)

    if args.num is not None:
        if args.persona is None:
            parser.error("--num requires --persona (test, all, or a domain name)")
        personas = personas[: args.num]

    print(f"Expanding {len(personas)} persona(s)\n")

    with ThreadPoolExecutor(max_workers=len(personas)) as executor:
        futures = {executor.submit(run_one, p): p for p in personas}
        for future in as_completed(futures):
            persona, output_path = future.result()
            print(f"  {persona}\n  -> {output_path}\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
