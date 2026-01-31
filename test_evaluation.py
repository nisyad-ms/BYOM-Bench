#!/usr/bin/env python
"""Test evaluation - Run agent dialogue and scoring on generated task.

Usage:
    python test_evaluation.py                     # Full-context agent, latest session/task
    python test_evaluation.py --no-context        # No-context agent (baseline)
    python test_evaluation.py --session <file>    # Use specific session
    python test_evaluation.py --task <file>       # Use specific task
    python test_evaluation.py --max-agent-turns 5
"""

import argparse
import json
import sys
from pathlib import Path

from utils import (
    add_file_logging,
    extract_session_id,
    extract_task_num,
    get_eval_path,
    get_latest_session,
    get_latest_task_for_session,
    setup_logging,
    validate_task_session_match,
)

logger = setup_logging("evaluation")


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session file (default: latest)")
    parser.add_argument("--task", type=str, default=None,
                        help="Path to task file (default: latest for session)")
    parser.add_argument("--no-context", action="store_true",
                        help="Evaluate agent without conversation history (should score poorly)")
    parser.add_argument("--max-agent-turns", type=int, default=10,
                        help="Maximum agent turns in dialogue")
    args = parser.parse_args()

    # Find session file
    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            logger.error(f"Session file not found: {session_path}")
            sys.exit(1)
    else:
        session_path = get_latest_session()
        if session_path is None:
            logger.error("No session files found in outputs/conversation/")
            logger.info("Run test_data_generation.py first")
            sys.exit(1)
        logger.info(f"Using latest session: {session_path.name}")

    # Extract session ID
    session_id = extract_session_id(session_path)
    if session_id is None:
        logger.error(f"Could not extract session ID from: {session_path}")
        sys.exit(1)

    # Find task file
    if args.task:
        task_path = Path(args.task)
        if not task_path.exists():
            logger.error(f"Task file not found: {task_path}")
            sys.exit(1)
        # Validate task matches session
        if not validate_task_session_match(session_path, task_path):
            task_session_id = extract_session_id(task_path)
            logger.error(f"Task session ID ({task_session_id}) does not match session ID ({session_id})")
            logger.info("Use matching session and task files, or let the script auto-detect")
            sys.exit(1)
    else:
        task_path = get_latest_task_for_session(session_id)
        if task_path is None:
            logger.error(f"No task files found for session: {session_id}")
            logger.info("Run test_task_generation.py first")
            sys.exit(1)
        logger.info(f"Using latest task: {task_path.name}")

    # Determine output path
    agent_type = "nocontext" if args.no_context else "context"
    task_num = extract_task_num(task_path) or 1
    output_path = get_eval_path(session_id, task_num, agent_type)

    from persona_gym.evaluation_multisession import run_evaluation
    from persona_gym.schemas import EvaluationTask, MultiSessionOutput

    # Load data and task
    logger.info("=" * 60)
    logger.info("EVALUATION TEST")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Session: {session_path.name}")
    logger.info(f"  Task: {task_path.name}")
    logger.info(f"  Mode: {'No-Context Agent' if args.no_context else 'Full-Context Agent'}")
    logger.info(f"  Max Agent Turns: {args.max_agent_turns}")
    logger.info(f"  Output: {output_path}")
    logger.info("")

    with open(session_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)
    logger.info(f"Loaded {len(data.sessions)} sessions")

    with open(task_path, "r", encoding="utf-8") as f:
        raw_task = json.load(f)
    eval_task = EvaluationTask.from_dict(raw_task)
    logger.info(f"Loaded task: {eval_task.evaluation_event.event}")
    logger.info("")

    # Configure agent mode
    include_history = not args.no_context

    # Run evaluation
    logger.info("--- Running Evaluation ---")
    logger.info("")
    result = run_evaluation(
        data,
        max_agent_turns=args.max_agent_turns,
        include_history=include_history,
        eval_task=eval_task,
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
    logger.info("")
    logger.info("Turn Analysis:")
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
