#!/usr/bin/env python
"""Test task generation - Generate and inspect evaluation tasks without running full evaluation.

Usage:
    # Generate task from existing multi-session data
    python test_task_generation.py

    # Save task to file for inspection
    python test_task_generation.py --save

    # Use a specific input file
    python test_task_generation.py --input outputs/my_data.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and inspect evaluation tasks"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/test_multisession_output.json",
        help="Path to multi-session output JSON file"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save generated task to file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: derived from input)"
    )
    parser.add_argument(
        "--num-stale-traps",
        type=int,
        default=2,
        help="Number of stale preferences to include as traps"
    )
    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run test_multisession.py first to generate multi-session data")
        sys.exit(1)

    # Import after arg parsing to avoid slow startup
    from persona_gym.schemas import MultiSessionOutput
    from persona_gym.task_generators import generate_evaluation_task

    # Load multi-session data
    logger.info(f"Loading multi-session data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions with {len(data.timeline.preferences)} preferences")

    # Display current vs stale preferences
    current_prefs = data.get_current_preferences()
    stale_prefs = data.get_superseded_preferences()
    print("\n" + "=" * 60)
    print("PREFERENCE ANALYSIS")
    print("=" * 60)
    print(f"\nCurrent preferences ({len(current_prefs)}):")
    for p in current_prefs:
        print(f"  [{p.preference_id}] {p.fact[:80]}...")

    print(f"\nStale/superseded preferences ({len(stale_prefs)}):")
    for p in stale_prefs:
        print(f"  [{p.preference_id}] {p.fact[:80]}... -> superseded by {p.superseded_by}")

    # Generate evaluation task
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION TASK")
    print("=" * 60)
    task = generate_evaluation_task(data, num_stale_traps=args.num_stale_traps)

    # Display task details
    print("\n" + "-" * 40)
    print("EVALUATION EVENT")
    print("-" * 40)
    print(f"Date: {task.evaluation_event.date}")
    print(f"Event: {task.evaluation_event.event}")
    print(f"Task: {task.evaluation_event.task}")
    print(f"Context: {task.evaluation_event.context}")
    print(f"Completion Criteria: {task.evaluation_event.completion_criteria}")
    print(f"Required Preferences: {task.evaluation_event.required_preferences}")

    print("\n" + "-" * 40)
    print("USER PROMPT")
    print("-" * 40)
    print(task.user_prompt)

    print("\n" + "-" * 40)
    print("EVALUATION RUBRIC")
    print("-" * 40)
    print(f"Required Preferences: {task.rubric.required_preferences}")
    print(f"Completion Criteria: {task.rubric.completion_criteria}")
    print(f"\nExpected Behaviors ({len(task.rubric.expected_behaviors)}):")
    for b in task.rubric.expected_behaviors:
        print(f"  • {b[:100]}...")
    print(f"\nTrap Behaviors ({len(task.rubric.trap_behaviors)}):")
    for b in task.rubric.trap_behaviors:
        print(f"  ⚠ {b[:100]}...")

    print("\n" + "-" * 40)
    print("PERSONA SUMMARY (for user simulator)")
    print("-" * 40)
    print(task.persona_summary)

    # Save if requested
    if args.save:
        output_path = args.output or input_path.with_name(
            input_path.stem.replace("_output", "_task") + ".json"
        )
        logger.info(f"Saving task to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n✓ Task saved to {output_path}")

    print("\n" + "=" * 60)
    print("TASK GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
