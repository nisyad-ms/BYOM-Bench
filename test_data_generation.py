#!/usr/bin/env python
"""Test data generation - Generate multi-session conversation data with preference evolution.

Usage:
    python test_data_generation.py
    python test_data_generation.py --sessions 3
    python test_data_generation.py --persona "A software engineer at Google"
"""

import argparse
import json
import time

from utils import add_file_logging, create_session_dir, get_session_path, setup_logging

logger = setup_logging("data_generation")

DEFAULT_PERSONA = """
A senior software engineer working at Microsoft.
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Test multi-session data generation")
    parser.add_argument("--sessions", type=int, default=2, help="Number of sessions")
    parser.add_argument("--persona", type=str, default=None, help="Custom persona description")
    args = parser.parse_args()

    from memory_gym.data_generators import MultiSessionGenerator

    persona = args.persona or DEFAULT_PERSONA
    session_dir = create_session_dir()
    add_file_logging(logger, session_dir)
    output_path = get_session_path(session_dir)

    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=args.sessions,
    )

    result = generator.generate_multi_session()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
