"""
Evaluation Runner for Multi-Session Preference Recall.

Orchestrates the complete evaluation flow:
1. Load multi-session data
2. Generate evaluation task (via orchestrator)
3. Run dialogue between agent and user simulator
4. Evaluate with judge
"""

import json
import logging
from pathlib import Path
from typing import Any

from persona_gym.agents import ContextAwareAgent, NoContextAgent
from persona_gym.client import AsyncLLMPool, LLMClient
from persona_gym.schemas import (
    EvaluationTask,
    MultiSessionEvaluationResult,
    MultiSessionOutput,
)
from persona_gym.task_generators import EvaluationTaskGenerator

from .judge import MultiSessionJudge
from .user_simulator import MultiSessionUserSimulator

logger = logging.getLogger(__name__)


def run_evaluation(
    multisession_data: MultiSessionOutput,
    agent_system_prompt: str | None = None,
    max_agent_turns: int = 10,
    include_history: bool = True,
    client: LLMClient | None = None,
    eval_task: EvaluationTask | None = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    This function:
    1. Generates an evaluation task (or uses provided one)
    2. Runs a dialogue between the agent and user simulator
    3. Evaluates the dialogue with the judge

    Args:
        multisession_data: Output from MultiSessionGenerator
        agent_system_prompt: Custom system prompt for the agent being evaluated.
            If None, uses a generic helpful assistant prompt.
        max_agent_turns: Maximum agent turns before ending (default 10)
        include_history: Whether to include conversation history in agent context.
            Set to False for no-context agent evaluation.
        client: Shared LLM client. If None, creates a new one.
        eval_task: Pre-generated evaluation task. If None, generates a new one.

    Returns:
        MultiSessionEvaluationResult with scores and analysis
    """
    client = client or LLMClient()

    # Step 1: Generate or use provided evaluation task
    if eval_task is None:
        logger.info("Generating evaluation task...")
        task_generator = EvaluationTaskGenerator(client)
        eval_task = task_generator.generate(multisession_data)
        logger.info(f"Generated task: {eval_task.evaluation_event.event}")
    else:
        logger.info(f"Using provided task: {eval_task.evaluation_event.event}")

    # Step 2: Run dialogue
    logger.info("Running evaluation dialogue...")
    conversation = run_dialogue(
        eval_task,
        multisession_data,
        agent_system_prompt,
        max_agent_turns,
        include_history,
        client,
    )
    logger.info(f"Dialogue completed: {len(conversation)} turns")

    # Step 3: Evaluate with judge
    logger.info("Evaluating dialogue with judge...")
    judge = MultiSessionJudge(client)
    result = judge.evaluate(eval_task, conversation)
    logger.info(f"Evaluation complete. Preference: {result.preference_score:.2f}, Efficiency: {result.efficiency_score:.2f}")

    return result


def run_dialogue(
    eval_task: EvaluationTask,
    multisession_data: MultiSessionOutput,
    agent_system_prompt: str | None,
    max_agent_turns: int,
    include_history: bool,
    client: LLMClient,
) -> list[dict[str, str]]:
    """Run the evaluation dialogue between agent and user simulator.

    Args:
        eval_task: The evaluation task
        multisession_data: Source data (used to build agent context)
        agent_system_prompt: Agent's system prompt
        max_agent_turns: Maximum agent turns
        include_history: Whether to include conversation history in agent context
        client: LLM client

    Returns:
        Conversation transcript
    """
    user_sim = MultiSessionUserSimulator(eval_task, client)

    if include_history:
        agent = ContextAwareAgent(client)
    else:
        agent = NoContextAgent(client)
    agent.build_context(multisession_data)

    conversation: list[dict[str, str]] = []

    user_message = user_sim.get_initial_message()
    conversation.append({"role": "user", "content": user_message})
    logger.debug(f"User: {user_message[:100]}...")

    agent_turns = 0
    while agent_turns < max_agent_turns:
        agent_response = agent.respond(conversation)
        conversation.append({"role": "assistant", "content": agent_response})
        agent_turns += 1
        logger.debug(f"Agent: {agent_response[:100]}...")

        if user_sim.should_end_conversation(agent_response, conversation):
            logger.info(f"Conversation ended naturally after {agent_turns} agent turns")
            break

        if agent_turns >= max_agent_turns:
            logger.warning(f"Conversation hit max agent turns limit ({max_agent_turns})")
            break

        user_response = user_sim.respond(agent_response, conversation)
        conversation.append({"role": "user", "content": user_response})
        logger.debug(f"User: {user_response[:100]}...")

        if user_sim.should_end_conversation("", conversation):
            logger.info(f"User ended conversation after {agent_turns} agent turns")
            break

    return conversation


def run_evaluation_from_file(
    data_path: str | Path,
    agent_system_prompt: str | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> MultiSessionEvaluationResult:
    """Run evaluation from a saved multi-session JSON file.

    Args:
        data_path: Path to the multi-session output JSON
        agent_system_prompt: Custom agent system prompt
        output_path: Optional path to save results
        **kwargs: Additional arguments passed to run_evaluation

    Returns:
        Evaluation result
    """
    data_path = Path(data_path)

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    multisession_data = MultiSessionOutput.from_dict(data)

    # Run evaluation
    result = run_evaluation(
        multisession_data,
        agent_system_prompt=agent_system_prompt,
        **kwargs,
    )

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    return result


# CLI entry point
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run multi-session preference recall evaluation"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to multi-session output JSON",
    )
    parser.add_argument(
        "--output",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=10,
        help="Maximum agent turns (default: 10)",
    )
    args = parser.parse_args()

    result = run_evaluation_from_file(
        args.input,
        output_path=args.output,
        max_agent_turns=args.max_agent_turns,
    )

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total Turns: {result.total_turns}")
    print(f"Correction Turns: {result.correction_turns}")
    print(f"Efficiency Score: {result.efficiency_score:.2f}")
    print(f"Preference Score: {result.preference_score:.2f}")
    print(f"\nReasoning: {result.reasoning}")


def _run_single_evaluation_with_client(
    client: LLMClient,
    context: dict,
) -> MultiSessionEvaluationResult:
    """Run a single evaluation using provided client (for parallel execution).

    Args:
        client: LLM client to use
        context: Dict containing multisession_data, eval_task, include_history, max_agent_turns

    Returns:
        MultiSessionEvaluationResult
    """
    return run_evaluation(
        multisession_data=context["multisession_data"],
        eval_task=context["eval_task"],
        include_history=context["include_history"],
        max_agent_turns=context["max_agent_turns"],
        client=client,
    )


async def run_evaluations_parallel(
    contexts: list[dict],
) -> list[MultiSessionEvaluationResult]:
    """Run multiple evaluations in parallel across deployments.

    Args:
        contexts: List of dicts, each containing:
            - multisession_data: MultiSessionOutput
            - eval_task: EvaluationTask
            - include_history: bool
            - max_agent_turns: int

    Returns:
        List of MultiSessionEvaluationResult objects
    """
    pool = AsyncLLMPool()

    results = await pool.run_parallel(
        items=contexts,
        func=_run_single_evaluation_with_client,
    )

    return results
