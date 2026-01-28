#!/usr/bin/env python
"""Run the full PersonaGym pipeline end-to-end.

Usage:
    # Full pipeline from scratch
    python test_pipeline.py

    # Skip data generation, use existing data
    python test_pipeline.py --use-existing-data outputs/data_generation_output.json

    # Compare full-context vs no-context agents
    python test_pipeline.py --compare-agents

    # Custom configuration
    python test_pipeline.py --sessions 3 --max-turns 8
"""

import argparse
import json
from pathlib import Path
from typing import Any

from utils import add_file_logging, setup_logging

logger = setup_logging("pipeline")

# Default test persona
DEFAULT_PERSONA = """
A 32-year-old software engineer at a mid-size tech company who has been feeling
stagnant in their career. They have 8 years of experience in backend development
but are curious about transitioning to machine learning. They're methodical,
prefer structured learning, and are budget-conscious. They live in Seattle with
their partner and enjoy hiking on weekends.
""".strip()


def run_data_generation(args: argparse.Namespace) -> Path:
    """Run data generation stage."""
    from persona_gym.data_generators import MultiSessionGenerator

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA GENERATION")
    logger.info("=" * 60)
    logger.info("")

    persona = args.persona or DEFAULT_PERSONA
    logger.info("Persona:")
    for line in persona.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    generator = MultiSessionGenerator(
        persona=persona,
        num_sessions=args.sessions,
    )

    result = generator.generate_multi_session()

    logger.info(f"Generated {len(result.sessions)} sessions")
    logger.info(f"{len(result.timeline.preferences)} total preferences")
    logger.info(f"{len(result.timeline.get_active_preferences())} active, "
                f"{len(result.get_superseded_preferences())} superseded")
    logger.info("")

    # Log preferences
    logger.info("Preferences:")
    for pref_id, pref in result.timeline.preferences.items():
        status = "ACTIVE" if pref.is_active else f"-> {pref.superseded_by}"
        logger.info(f"  [{pref_id}] {status}")
        logger.info(f"    {pref.fact}")
    logger.info("")

    # Log conversations
    for session in result.sessions:
        logger.info(f"Session {session.session_id} - {session.life_event.event}")
        logger.info("-" * 40)
        for turn in session.conversation:
            role = turn['role'].upper()
            logger.info(f"  [{role}]:")
            for line in turn['content'].split('\n'):
                logger.info(f"    {line}")
            logger.info("")
        logger.info("")

    # Save
    output_path = Path(args.output_dir) / "data_generation_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to: {output_path}")

    return output_path


def run_task_generation(data_path: Path, args: argparse.Namespace) -> tuple[Path, Any]:
    """Run task generation stage."""
    from persona_gym.schemas import MultiSessionOutput
    from persona_gym.task_generators import generate_evaluation_task

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: TASK GENERATION")
    logger.info("=" * 60)
    logger.info("")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = MultiSessionOutput.from_dict(raw_data)

    task = generate_evaluation_task(data)

    logger.info("Evaluation Event:")
    logger.info(f"  {task.evaluation_event.event}")
    logger.info("")
    logger.info("Task:")
    for line in task.evaluation_event.task.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    logger.info(f"Required Preferences: {task.rubric.required_preferences}")
    logger.info("")
    logger.info("User Prompt:")
    for line in task.user_prompt.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    # Save
    output_path = Path(args.output_dir) / "task_generation_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to: {output_path}")

    return output_path, data


def run_evaluation(data: Any, args: argparse.Namespace, no_context: bool = False) -> dict:
    """Run evaluation stage."""
    from persona_gym.evaluation_multisession import run_evaluation as eval_fn

    mode = "NO-CONTEXT" if no_context else "FULL-CONTEXT"
    logger.info("")
    logger.info(f"--- Running {mode} Agent Evaluation ---")
    logger.info("")

    if no_context:
        include_history = False
        output_name = "evaluation_output_no_context.json"
    else:
        include_history = True
        output_name = "evaluation_output.json"

    result = eval_fn(
        data,
        max_turns=args.max_turns,
        include_history=include_history,
    )

    logger.info("Scores:")
    logger.info(f"  Final Score: {result.final_score:.2f}")
    logger.info(f"  Preference Score: {result.preference_score:.2f}")
    logger.info(f"  Efficiency Score: {result.efficiency_score:.2f}")
    logger.info(f"  Stale Penalty: {result.stale_penalty:.2f}")
    logger.info("")
    logger.info(f"Task Completed: {result.task_completed}")
    logger.info(f"Total Turns: {result.total_turns}")
    logger.info("")

    # Log full conversation
    logger.info("Conversation:")
    logger.info("-" * 40)
    for i, turn in enumerate(result.conversation):
        role = turn["role"].upper()
        logger.info(f"[Turn {i+1}] {role}:")
        for line in turn["content"].split('\n'):
            logger.info(f"  {line}")
        logger.info("")
    logger.info("-" * 40)
    logger.info("")

    # Log reasoning
    logger.info("Judge Reasoning:")
    for line in result.reasoning.split('\n'):
        logger.info(f"  {line}")
    logger.info("")

    # Save
    output_path = Path(args.output_dir) / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to: {output_path}")

    return {
        "mode": mode,
        "final_score": result.final_score,
        "preference_score": result.preference_score,
        "efficiency_score": result.efficiency_score,
        "task_completed": result.task_completed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PersonaGym pipeline")

    # Stage control
    parser.add_argument("--use-existing-data", type=str, default=None,
                        help="Skip data generation, use existing file")
    parser.add_argument("--compare-agents", action="store_true",
                        help="Run both full-context and no-context evaluations")
    parser.add_argument("--no-context", action="store_true",
                        help="Only run no-context agent evaluation")

    # Data generation params
    parser.add_argument("--sessions", type=int, default=2, help="Number of sessions")
    parser.add_argument("--persona", type=str, default=None, help="Custom persona")

    # Evaluation params
    parser.add_argument("--max-turns", type=int, default=5, help="Max agent turns")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PERSONAGYM PIPELINE")
    logger.info("=" * 60)

    # Stage 1: Data Generation
    if args.use_existing_data:
        data_path = Path(args.use_existing_data)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
        logger.info(f"Using existing data: {data_path}")
    else:
        data_path = run_data_generation(args)

    # Stage 2: Task Generation
    task_path, data = run_task_generation(data_path, args)

    # Stage 3: Evaluation
    logger.info("=" * 60)
    logger.info("STAGE 3: EVALUATION")
    logger.info("=" * 60)

    results = []

    if args.compare_agents:
        results.append(run_evaluation(data, args, no_context=False))
        results.append(run_evaluation(data, args, no_context=True))
    elif args.no_context:
        results.append(run_evaluation(data, args, no_context=True))
    else:
        results.append(run_evaluation(data, args, no_context=False))

    # Final Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    if len(results) > 1:
        logger.info("--- Agent Comparison ---")
        logger.info(f"{'Agent':<15} {'Final Score':<12} {'Pref Score':<12} {'Efficiency':<12}")
        logger.info("-" * 51)
        for r in results:
            logger.info(f"{r['mode']:<15} {r['final_score']:<12.2f} {r['preference_score']:<12.2f} {r['efficiency_score']:<12.2f}")

        diff = results[0]['final_score'] - results[1]['final_score']
        logger.info(f"Full-context agent scores {diff:.2f} points higher than no-context")
    else:
        r = results[0]
        logger.info(f"{r['mode']} Agent Final Score: {r['final_score']:.2f}")

    logger.info(f"Outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    add_file_logging(logger)
    main()
