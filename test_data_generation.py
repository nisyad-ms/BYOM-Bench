#!/usr/bin/env python
"""Test data generation - Generate multi-session conversation data with preference evolution.

Usage:
    python test_data_generation.py
    python test_data_generation.py --output outputs/my_data.json
    python test_data_generation.py --sessions 3
"""

import argparse
import json
from pathlib import Path

from utils import add_file_logging, setup_logging

logger = setup_logging("data_generation")

# Default test persona
DEFAULT_PERSONA = """
A senior software engineer working at Microsoft.
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Test multi-session data generation")
    parser.add_argument("--output", type=str, default="outputs/data_generation_output.json",
                        help="Output file path")
    parser.add_argument("--sessions", type=int, default=2, help="Number of sessions")
    parser.add_argument("--persona", type=str, default=None, help="Custom persona description")
    args = parser.parse_args()

    from persona_gym.data_generators import MultiSessionGenerator

    persona = args.persona or DEFAULT_PERSONA

    logger.info("=" * 60)
    logger.info("DATA GENERATION TEST")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Sessions: {args.sessions}")
    logger.info("")
    logger.info("Persona:")
    logger.info("-" * 40)
    for line in persona.split('\n'):
        logger.info(f"  {line}")
    logger.info("-" * 40)
    logger.info("")

    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=args.sessions,
    )

    result = generator.generate_multi_session()

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Persona ID: {result.persona_id}")
    logger.info(f"Total Sessions: {len(result.sessions)}")
    logger.info(f"Total Preferences: {len(result.timeline.preferences)}")
    logger.info(f"Active Preferences: {len(result.timeline.get_active_preferences())}")

    logger.info("")
    logger.info("--- Life Events ---")
    logger.info("")
    for event in result.life_events:
        logger.info(f"  [{event.session_id}] {event.date}: {event.event}")
        if event.context:
            logger.info(f"      Context: {event.context}")

    logger.info("")
    logger.info("--- Preference Timeline ---")
    logger.info("")
    for pref_id, pref in result.timeline.preferences.items():
        status = "ACTIVE" if pref.is_active else f"SUPERSEDED by {pref.superseded_by}"
        logger.info(f"  {pref_id}: [{pref.domain}] ({status})")
        logger.info(f"    {pref.fact}")
        logger.info("")

    logger.info("")
    logger.info("--- Sessions ---")
    for session in result.sessions:
        logger.info("")
        logger.info(f"  Session {session.session_id}:")
        logger.info(f"    Life Event: {session.life_event.event}")
        logger.info(f"    Active Prefs: {session.active_preference_ids}")
        if session.evolved_preference_ids:
            logger.info(f"    Evolved: {session.evolved_preference_ids}")
        logger.info("")
        logger.info(f"    Conversation ({len(session.conversation)} turns):")
        logger.info("    " + "-" * 50)
        for turn in session.conversation:
            role = turn['role'].upper()
            content = turn['content']
            logger.info(f"    [{role}]:")
            for line in content.split('\n'):
                logger.info(f"      {line}")
            logger.info("")
        logger.info("    " + "-" * 50)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("")
    logger.info(f"Output saved to: {output_path}")
    logger.info("")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
