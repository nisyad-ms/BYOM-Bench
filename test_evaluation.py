#!/usr/bin/env python
"""Test evaluation - Run agent dialogue and scoring on generated task.

Usage:
    python test_evaluation.py --input outputs/data_generation_output.json
    python test_evaluation.py --no-context  # Test agent without memory
    python test_evaluation.py --max-turns 10
"""

import argparse
import json
import sys
from pathlib import Path

from utils import add_file_logging, setup_logging

logger = setup_logging("evaluation")


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system")
    parser.add_argument("--input", type=str, default="outputs/data_generation_output.json",
                        help="Path to multi-session data JSON file")
    parser.add_argument("--output", type=str, default="outputs/evaluation_output.json",
                        help="Output file path")
    parser.add_argument("--no-context", action="store_true",
                        help="Evaluate agent without conversation history (should score poorly)")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum agent turns in dialogue")
    args = parser.parse_args()

    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run test_data_generation.py first")
        sys.exit(1)

    from persona_gym.evaluation_multisession import run_evaluation
    from persona_gym.schemas import MultiSessionOutput

    # Load data
    logger.info("=" * 60)
    logger.info("EVALUATION TEST")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Mode: {'No-Context Agent' if args.no_context else 'Full-Context Agent'}")
    logger.info(f"  Max Turns: {args.max_turns}")
    logger.info("")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions")
    logger.info("")

    # Configure agent mode
    if args.no_context:
        include_history = False
        output_path = Path(args.output.replace(".json", "_no_context.json"))
    else:
        include_history = True
        output_path = Path(args.output)

    # Run evaluation
    logger.info("--- Running Evaluation ---")
    logger.info("")
    result = run_evaluation(
        data,
        max_turns=args.max_turns,
        include_history=include_history,
    )

    # Log results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Scores:")
    logger.info(f"  Final Score: {result.final_score:.2f}")
    logger.info(f"  Preference Score: {result.preference_score:.2f}")
    logger.info(f"  Efficiency Score: {result.efficiency_score:.2f}")
    logger.info(f"  Stale Penalty: {result.stale_penalty:.2f}")
    logger.info("")
    logger.info("Turn Analysis:")
    logger.info(f"  Task Completed: {result.task_completed}")
    logger.info(f"  Total Turns: {result.total_turns}")
    logger.info(f"  Productive Turns: {result.productive_turns}")
    logger.info(f"  Clarifying Turns: {result.clarifying_turns}")
    logger.info(f"  Correction Turns: {result.correction_turns}")
    logger.info("")
    logger.info("Judge Reasoning:")
    logger.info("-" * 40)
    for line in result.reasoning.split('\n'):
        logger.info(f"  {line}")
    logger.info("-" * 40)
    logger.info("")

    logger.info("--- Full Conversation ---")
    logger.info("")
    for i, turn in enumerate(result.conversation):
        role = turn["role"].upper()
        content = turn["content"]
        logger.info(f"[Turn {i+1}] {role}:")
        for line in content.split('\n'):
            logger.info(f"  {line}")
        logger.info("")
    logger.info("-" * 40)
    logger.info("")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")
    logger.info("")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
