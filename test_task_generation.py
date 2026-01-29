#!/usr/bin/env python
"""Test task generation - Generate evaluation tasks from multi-session data.

Usage:
    python test_task_generation.py
    python test_task_generation.py --input outputs/data_generation_output.json
    python test_task_generation.py --save --output outputs/task_generation_output.json
"""

import argparse
import json
import sys
from pathlib import Path

from utils import add_file_logging, setup_logging

logger = setup_logging("task_generation")


def main():
    parser = argparse.ArgumentParser(description="Test evaluation task generation")
    parser.add_argument("--input", type=str, default="outputs/data_generation_output.json",
                        help="Path to multi-session data JSON file")
    parser.add_argument("--save", action="store_true", help="Save generated task to file")
    parser.add_argument("--output", type=str, default="outputs/task_generation_output.json",
                        help="Output file path")
    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run test_data_generation.py first")
        sys.exit(1)

    from persona_gym.schemas import MultiSessionOutput
    from persona_gym.task_generators import generate_evaluation_task

    # Load multi-session data
    logger.info(f"Loading multi-session data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions with {len(data.timeline.preferences)} preferences")
    logger.info("")

    # Display current vs stale preferences
    current_prefs = data.get_current_preferences()
    stale_prefs = data.get_superseded_preferences()

    logger.info("=" * 60)
    logger.info("PREFERENCE ANALYSIS")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Current preferences ({len(current_prefs)}):")
    logger.info("")
    for p in current_prefs:
        logger.info(f"  [{p.preference_id}] [{p.domain}]")
        logger.info(f"    {p.fact}")
        logger.info("")

    logger.info(f"Stale/superseded preferences ({len(stale_prefs)}):")
    logger.info("")
    for p in stale_prefs:
        logger.info(f"  [{p.preference_id}] [{p.domain}] -> superseded by {p.superseded_by}")
        logger.info(f"    {p.fact}")
        logger.info("")

    # Generate evaluation task
    logger.info("=" * 60)
    logger.info("GENERATING EVALUATION TASK")
    logger.info("=" * 60)
    logger.info("")
    task = generate_evaluation_task(data)

    # Display task details
    logger.info("-" * 40)
    logger.info("EVALUATION EVENT")
    logger.info("-" * 40)
    logger.info("")
    logger.info(f"Date: {task.evaluation_event.date}")
    logger.info("")
    logger.info("Event:")
    logger.info(f"  {task.evaluation_event.event}")
    logger.info("")
    logger.info("Task:")
    for line in task.evaluation_event.task.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info("Context:")
    for line in task.evaluation_event.context.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info("Completion Criteria:")
    for line in task.evaluation_event.completion_criteria.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info(f"Required Preferences: {task.evaluation_event.required_preferences}")
    logger.info("")

    logger.info("-" * 40)
    logger.info("USER PROMPT")
    logger.info("-" * 40)
    logger.info("")
    for line in task.user_prompt.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    logger.info("-" * 40)
    logger.info("EVALUATION RUBRIC")
    logger.info("-" * 40)
    logger.info("")
    logger.info(f"Required Preferences: {task.rubric.required_preferences}")
    logger.info("")
    logger.info("Completion Criteria:")
    for line in task.rubric.completion_criteria.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info(f"Expected Behaviors ({len(task.rubric.expected_behaviors)}):")
    for b in task.rubric.expected_behaviors:
        logger.info(f"  • {b}")
    logger.info("")
    logger.info(f"Trap Behaviors ({len(task.rubric.trap_behaviors)}):")
    for b in task.rubric.trap_behaviors:
        logger.info(f"  ⚠ {b}")
    logger.info("")

    logger.info("-" * 40)
    logger.info("PERSONA SUMMARY (for user simulator)")
    logger.info("-" * 40)
    logger.info("")
    for line in task.persona_summary.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    # Save if requested
    if args.save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Task saved to {output_path}")
        logger.info("")

    logger.info("=" * 60)
    logger.info("TASK GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
