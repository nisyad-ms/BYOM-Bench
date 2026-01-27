"""
Test multi-session evaluation system.

This test:
1. Loads previously generated multi-session data
2. Generates an evaluation task
3. Runs a short evaluation dialogue
4. Evaluates with the judge
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_evaluation_components():
    """Test individual components of the evaluation system."""
    from persona_gym.client import LLMClient
    from persona_gym.schemas import MultiSessionOutput

    # Load test data
    data_path = Path("outputs/test_multisession_output.json")
    if not data_path.exists():
        logger.error(f"Test data not found at {data_path}")
        logger.info("Run test_multisession.py first to generate test data")
        return

    with open(data_path) as f:
        data = json.load(f)

    multisession_data = MultiSessionOutput.from_dict(data)
    logger.info(f"Loaded multi-session data: {len(multisession_data.sessions)} sessions")

    # Check preferences
    current_prefs = multisession_data.get_current_preferences()
    stale_prefs = multisession_data.get_superseded_preferences()
    logger.info(f"Current preferences: {len(current_prefs)}")
    logger.info(f"Stale preferences: {len(stale_prefs)}")

    for p in current_prefs:
        logger.info(f"  - {p.preference_id}: {p.fact[:60]}...")

    for p in stale_prefs:
        logger.info(f"  - [STALE] {p.preference_id}: {p.fact[:40]}... -> {p.superseded_by}")

    # Test orchestrator
    logger.info("\n--- Testing Orchestrator ---")
    from persona_gym.evaluation_multisession.orchestrator import EvaluationOrchestrator

    client = LLMClient()
    orchestrator = EvaluationOrchestrator(client)

    eval_task = orchestrator.generate_evaluation_task(multisession_data, num_stale_traps=2)

    logger.info(f"Generated evaluation event: {eval_task.evaluation_event.event}")
    logger.info(f"User prompt: {eval_task.user_prompt[:100]}...")
    logger.info(f"Current prefs in rubric: {len(eval_task.rubric.current_preferences)}")
    logger.info(f"Stale prefs in rubric: {len(eval_task.rubric.stale_preferences)}")

    # Test user simulator
    logger.info("\n--- Testing User Simulator ---")
    from persona_gym.evaluation_multisession.user_simulator import MultiSessionUserSimulator

    user_sim = MultiSessionUserSimulator(eval_task, client)
    initial_msg = user_sim.get_initial_message()
    logger.info(f"Initial user message: {initial_msg}")

    # Simulate one turn
    test_agent_msg = "I'd be happy to help you with that! What specific details would you like me to consider?"
    test_history = [
        {"role": "user", "content": initial_msg},
        {"role": "assistant", "content": test_agent_msg},
    ]

    user_response = user_sim.respond(test_agent_msg, test_history)
    logger.info(f"User response: {user_response}")

    return eval_task, client


def test_full_evaluation(no_context: bool = False):
    """Test full evaluation flow.

    Args:
        no_context: If True, evaluate an agent without access to conversation history.
            This agent should score poorly since it can't recall preferences.
    """
    from persona_gym.evaluation_multisession.runner import run_evaluation_from_file

    data_path = Path("outputs/test_multisession_output.json")
    if not data_path.exists():
        logger.error(f"Test data not found at {data_path}")
        return

    if no_context:
        logger.info("\n--- Running No-Context Agent Evaluation ---")
        output_path = Path("outputs/evaluation_result_no_context.json")
        # System prompt that explicitly has NO access to history
        agent_system_prompt = """You are a helpful AI assistant. This is your first conversation with this user - you have no prior context or history with them.

Be helpful and ask clarifying questions as needed since you don't know anything about the user's preferences, background, or previous discussions."""
        include_history = False
    else:
        logger.info("\n--- Running Full Evaluation ---")
        output_path = Path("outputs/evaluation_result.json")
        agent_system_prompt = None  # Uses default with history
        include_history = True

    result = run_evaluation_from_file(
        data_path,
        agent_system_prompt=agent_system_prompt,
        output_path=output_path,
        max_turns=5,  # Short test
        num_stale_traps=2,
        include_history=include_history,
    )

    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Task Completed: {result.task_completed}")
    logger.info(f"Total Turns: {result.total_turns}")
    logger.info(f"Productive Turns: {result.productive_turns}")
    logger.info(f"Clarifying Turns: {result.clarifying_turns}")
    logger.info(f"Correction Turns: {result.correction_turns}")
    logger.info(f"Efficiency Score: {result.efficiency_score:.2f}")
    logger.info(f"Preference Score: {result.preference_score:.2f}")
    logger.info(f"Stale Penalty: {result.stale_penalty:.2f}")
    logger.info(f"Final Score: {result.final_score:.2f}")
    logger.info(f"\nReasoning: {result.reasoning}")

    # Log conversation
    logger.info(f"\n--- Conversation ({len(result.conversation)} turns) ---")
    for turn in result.conversation:
        role = turn["role"].upper()
        content = turn["content"][:100] + ("..." if len(turn["content"]) > 100 else "")
        logger.info(f"{role}: {content}")

    return result


if __name__ == "__main__":
    import sys

    if "--no-context" in sys.argv:
        test_full_evaluation(no_context=True)
    elif "--full" in sys.argv:
        test_full_evaluation(no_context=False)
    else:
        # Quick component test
        test_evaluation_components()
        logger.info("\nTo run full evaluation, use: python test_evaluation.py --full")
        logger.info("To test no-context agent, use: python test_evaluation.py --no-context")
