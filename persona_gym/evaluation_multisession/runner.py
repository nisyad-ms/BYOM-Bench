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

from persona_gym.client import LLMClient
from persona_gym.schemas import (
    EvaluationTask,
    MultiSessionEvaluationResult,
    MultiSessionOutput,
)

from .judge import MultiSessionJudge
from .orchestrator import EvaluationOrchestrator
from .user_simulator import MultiSessionUserSimulator

logger = logging.getLogger(__name__)


def run_evaluation(
    multisession_data: MultiSessionOutput,
    agent_system_prompt: str | None = None,
    max_turns: int = 10,
    num_stale_traps: int = 2,
    include_history: bool = True,
    client: LLMClient | None = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    This function:
    1. Generates an evaluation task from the multi-session data
    2. Runs a dialogue between the agent and user simulator
    3. Evaluates the dialogue with the judge

    Args:
        multisession_data: Output from MultiSessionGenerator
        agent_system_prompt: Custom system prompt for the agent being evaluated.
            If None, uses a generic helpful assistant prompt.
        max_turns: Maximum agent turns before ending (default 10)
        num_stale_traps: Number of stale preferences to include as traps
        include_history: Whether to include conversation history in agent context.
            Set to False for no-context agent evaluation.
        client: Shared LLM client. If None, creates a new one.

    Returns:
        MultiSessionEvaluationResult with scores and analysis
    """
    client = client or LLMClient()

    # Step 1: Generate evaluation task
    logger.info("Generating evaluation task...")
    orchestrator = EvaluationOrchestrator(client)
    eval_task = orchestrator.generate_evaluation_task(
        multisession_data, num_stale_traps
    )
    logger.info(f"Generated task: {eval_task.evaluation_event.event}")

    # Step 2: Run dialogue
    logger.info("Running evaluation dialogue...")
    conversation = run_dialogue(
        eval_task,
        multisession_data,
        agent_system_prompt,
        max_turns,
        include_history,
        client,
    )
    logger.info(f"Dialogue completed: {len(conversation)} turns")

    # Step 3: Evaluate with judge
    logger.info("Evaluating dialogue with judge...")
    judge = MultiSessionJudge(client)
    result = judge.evaluate(eval_task, conversation)
    logger.info(f"Evaluation complete. Final score: {result.final_score:.2f}")

    return result


def run_dialogue(
    eval_task: EvaluationTask,
    multisession_data: MultiSessionOutput,
    agent_system_prompt: str | None,
    max_turns: int,
    include_history: bool,
    client: LLMClient,
) -> list[dict[str, str]]:
    """Run the evaluation dialogue between agent and user simulator.

    Args:
        eval_task: The evaluation task
        multisession_data: Source data (used to build agent context)
        agent_system_prompt: Agent's system prompt
        max_turns: Maximum agent turns
        include_history: Whether to include conversation history in agent context
        client: LLM client

    Returns:
        Conversation transcript
    """
    # Create user simulator
    user_sim = MultiSessionUserSimulator(eval_task, client)

    # Build agent context from training sessions
    agent_context = build_agent_context(multisession_data, agent_system_prompt, include_history)

    # Initialize conversation
    conversation: list[dict[str, str]] = []

    # User starts
    user_message = user_sim.get_initial_message()
    conversation.append({"role": "user", "content": user_message})
    logger.debug(f"User: {user_message[:100]}...")

    agent_turns = 0
    while agent_turns < max_turns:
        # Agent responds
        agent_response = get_agent_response(conversation, agent_context, client)
        conversation.append({"role": "assistant", "content": agent_response})
        agent_turns += 1
        logger.debug(f"Agent: {agent_response[:100]}...")

        # Check if user wants to end
        if user_sim.should_end_conversation(agent_response, conversation):
            logger.info(f"Conversation ended naturally after {agent_turns} agent turns")
            break

        # User responds
        user_response = user_sim.respond(agent_response, conversation)
        conversation.append({"role": "user", "content": user_response})
        logger.debug(f"User: {user_response[:100]}...")

        # Check again after user response
        if user_sim.should_end_conversation("", conversation):
            logger.info(f"User ended conversation after {agent_turns} agent turns")
            break

    if agent_turns >= max_turns:
        logger.warning(f"Conversation hit max turns limit ({max_turns})")

    return conversation


def build_agent_context(
    multisession_data: MultiSessionOutput,
    custom_system_prompt: str | None = None,
    include_history: bool = True,
) -> str:
    """Build the agent's system prompt with conversation history context.

    Args:
        multisession_data: The multi-session data (agent's "memory")
        custom_system_prompt: Optional custom system prompt
        include_history: Whether to include conversation history. If False,
            the agent has no context (simulates a no-memory agent).

    Returns:
        Complete system prompt for the agent
    """
    # If no history, just return the custom prompt (no-context agent)
    if not include_history:
        return custom_system_prompt or """You are a helpful AI assistant. This is your first conversation with this user. You have no prior context or history with them.

Be helpful and ask clarifying questions as needed since you don't know anything about the user's preferences."""

    base_prompt = custom_system_prompt or """You are a helpful AI assistant. You have access to conversation history with this user from previous sessions. Use this context to provide personalized recommendations and remember their preferences.

Be proactive about using what you know about the user - don't wait for them to remind you of their preferences."""

    # Format conversation history
    history_parts = []
    for session in multisession_data.sessions:
        session_header = f"\n--- Session {session.session_id}: {session.life_event.event} ({session.life_event.date}) ---"
        history_parts.append(session_header)

        for turn in session.conversation:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            history_parts.append(f"{role}: {content}")

    conversation_history = "\n".join(history_parts)

    return f"""{base_prompt}

## Previous Conversation History with This User

{conversation_history}

---

Now respond to the user's new message. Remember to use the context from previous conversations to personalize your response."""


def get_agent_response(
    conversation: list[dict[str, str]],
    system_prompt: str,
    client: LLMClient,
) -> str:
    """Get agent response using the LLM.

    Args:
        conversation: Current conversation history
        system_prompt: Agent's system prompt (includes context)
        client: LLM client

    Returns:
        Agent's response message
    """
    # Format conversation as a chat-style prompt
    messages_str = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in conversation
    )

    prompt = f"""Here is the current conversation:

{messages_str}

How do you respond as the assistant? Be helpful and personalized based on what you know about this user from previous conversations.

Respond with just your message (no "Assistant:" prefix)."""

    response = client.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=512,
        temperature=0.7,
    )

    return response.strip()


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
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agent turns (default: 10)",
    )
    parser.add_argument(
        "--stale-traps",
        type=int,
        default=2,
        help="Number of stale preferences to test (default: 2)",
    )

    args = parser.parse_args()

    result = run_evaluation_from_file(
        args.input,
        output_path=args.output,
        max_turns=args.max_turns,
        num_stale_traps=args.stale_traps,
    )

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Task Completed: {result.task_completed}")
    print(f"Total Turns: {result.total_turns}")
    print(f"Correction Turns: {result.correction_turns}")
    print(f"Efficiency Score: {result.efficiency_score:.2f}")
    print(f"Preference Score: {result.preference_score:.2f}")
    print(f"Stale Penalty: {result.stale_penalty:.2f}")
    print(f"Task Success: {result.task_success_score:.2f}")
    print(f"Final Score: {result.final_score:.2f}")
    print(f"\nReasoning: {result.reasoning}")
