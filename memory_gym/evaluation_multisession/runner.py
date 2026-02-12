"""
Evaluation Runner for Multi-Session Preference Recall.

Orchestrates the complete evaluation flow:
1. Load multi-session data
2. Generate evaluation task (via orchestrator)
3. Run dialogue between agent and user simulator
4. Evaluate with judge
"""

import re
from typing import Callable, Literal

from memory_gym.agents import ContextAwareAgent, FoundryMemoryAgent, NoContextAgent
from memory_gym.client import AsyncLLMPool, LLMClient, PooledLLMClient
from memory_gym.formatting import summarize_events
from memory_gym.schemas import (
    EvaluationTask,
    MultiSessionEvaluationResult,
    MultiSessionOutput,
)
from memory_gym.task_generators import EvaluationTaskGenerator

from .judge import MultiSessionJudge
from .user_simulator import MultiSessionUserSimulator


def _is_uncovered_empty(scratchpad: str) -> bool:
    match = re.search(r"UNCOVERED:\s*\[([^\]]*)\]", scratchpad)
    if match:
        return match.group(1).strip() == ""
    match = re.search(r"UNCOVERED:\s*$", scratchpad, flags=re.MULTILINE)
    return match is not None


def run_evaluation(
    multisession_data: MultiSessionOutput,
    max_agent_turns: int = 10,
    include_history: bool = True,
    client: LLMClient | PooledLLMClient | None = None,
    eval_task: EvaluationTask | None = None,
    agent_type: Literal["context", "nocontext", "foundry"] | None = None,
    memory_store_name: str | None = None,
    force_recreate_memory: bool = False,
    foundry_agent: FoundryMemoryAgent | None = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    This function:
    1. Generates an evaluation task (or uses provided one)
    2. Runs a dialogue between the agent and user simulator
    3. Evaluates the dialogue with the judge

    Args:
        multisession_data: Output from MultiSessionGenerator
        max_agent_turns: Maximum agent turns before ending (default 10)
        include_history: Whether to include conversation history in agent context.
            Set to False for no-context agent evaluation. Ignored if agent_type is set.
        client: Shared LLM client. If None, creates a new one.
        eval_task: Pre-generated evaluation task. If None, generates a new one.
        agent_type: Type of agent to use: "context", "nocontext", or "foundry".
            If None, uses include_history to determine (for backward compatibility).
        memory_store_name: Name of memory store for foundry agent (required if agent_type="foundry").
        force_recreate_memory: If True, recreate memory store from scratch (foundry only).

    Returns:
        MultiSessionEvaluationResult with scores and analysis
    """
    client = client or PooledLLMClient()

    if agent_type is None:
        agent_type = "context" if include_history else "nocontext"

    # Step 1: Generate or use provided evaluation task
    if eval_task is None:
        task_generator = EvaluationTaskGenerator(client)
        eval_task = task_generator.generate(multisession_data)

    # Step 2: Run dialogue
    conversation_with_scratchpads, clean_conversation = run_dialogue(
        eval_task,
        multisession_data,
        max_agent_turns,
        agent_type,
        client,
        memory_store_name,
        force_recreate_memory,
        foundry_agent,
    )

    # Step 3: Evaluate with judge (using clean conversation without scratchpads)
    judge = MultiSessionJudge(client)
    result = judge.evaluate(eval_task, clean_conversation)

    # Replace conversation with version that includes scratchpads for output
    result.conversation = conversation_with_scratchpads

    return result


def run_dialogue(
    eval_task: EvaluationTask,
    multisession_data: MultiSessionOutput,
    max_agent_turns: int,
    agent_type: Literal["context", "nocontext", "foundry"],
    client: LLMClient | PooledLLMClient,
    memory_store_name: str | None = None,
    force_recreate_memory: bool = False,
    foundry_agent: FoundryMemoryAgent | None = None,
) -> tuple[list[dict[str, str | None]], list[dict[str, str]]]:
    """Run the evaluation dialogue between agent and user simulator.

    Args:
        eval_task: The evaluation task
        multisession_data: Source data (used to build agent context)
        max_agent_turns: Maximum agent turns
        agent_type: Type of agent: "context", "nocontext", or "foundry"
        client: LLM client
        memory_store_name: Name of memory store for foundry agent
        force_recreate_memory: If True, recreate memory store from scratch (foundry only)

    Returns:
        Tuple of (conversation_with_scratchpads, clean_conversation)
        - conversation_with_scratchpads: List with "scratchpad" key on user turns (for output JSON)
        - clean_conversation: List without scratchpads (for judge)
    """
    user_sim = MultiSessionUserSimulator(eval_task, client)
    event_summaries = summarize_events(multisession_data, client)

    if agent_type == "foundry":
        if foundry_agent is not None:
            agent = foundry_agent
            agent.reset_conversation()
            agent.build_context(multisession_data, force_recreate=force_recreate_memory)
        else:
            if not memory_store_name:
                raise ValueError("memory_store_name required for foundry agent")
            agent = FoundryMemoryAgent(memory_store_name=memory_store_name)
            agent.build_context(multisession_data, force_recreate=force_recreate_memory)
    elif agent_type == "context":
        agent = ContextAwareAgent(client)
        agent.build_context(multisession_data, event_summaries=event_summaries)
    else:
        agent = NoContextAgent(client)
        agent.build_context(multisession_data)

    conversation_with_scratchpads: list[dict[str, str | None]] = []
    clean_conversation: list[dict[str, str]] = []

    user_message = user_sim.get_initial_message()
    conversation_with_scratchpads.append({"role": "user", "content": user_message})
    clean_conversation.append({"role": "user", "content": user_message})

    agent_turns = 0
    while agent_turns < max_agent_turns:
        agent_response = agent.respond(clean_conversation)
        conversation_with_scratchpads.append({"role": "assistant", "content": agent_response})
        clean_conversation.append({"role": "assistant", "content": agent_response})
        agent_turns += 1

        if agent_turns >= max_agent_turns:
            print(f"Conversation hit max agent turns limit ({max_agent_turns})")
            break

        user_response, scratchpad = user_sim.respond(clean_conversation)
        conversation_with_scratchpads.append({"role": "user", "content": user_response, "scratchpad": scratchpad})
        clean_conversation.append({"role": "user", "content": user_response})

        if scratchpad and _is_uncovered_empty(scratchpad):
            break

    return conversation_with_scratchpads, clean_conversation


def _run_single_evaluation_with_client(
    client: LLMClient | PooledLLMClient,
    context: dict,
) -> MultiSessionEvaluationResult:
    """Run a single evaluation using provided client (for parallel execution).

    Args:
        client: LLM client to use
        context: Dict containing multisession_data, eval_task, include_history, max_agent_turns,
                 and optionally agent_type, memory_store_name

    Returns:
        MultiSessionEvaluationResult
    """
    return run_evaluation(
        multisession_data=context["multisession_data"],
        eval_task=context["eval_task"],
        include_history=context.get("include_history", True),
        max_agent_turns=context["max_agent_turns"],
        client=client,
        agent_type=context.get("agent_type"),
        memory_store_name=context.get("memory_store_name"),
        force_recreate_memory=context.get("force_recreate_memory", False),
        foundry_agent=context.get("foundry_agent"),
    )


async def run_evaluations_parallel(
    contexts: list[dict],
    on_result: Callable[[int, dict, "MultiSessionEvaluationResult"], None] | None = None,
) -> list[MultiSessionEvaluationResult]:
    """Run multiple evaluations in parallel across deployments.

    Args:
        contexts: List of dicts, each containing:
            - multisession_data: MultiSessionOutput
            - eval_task: EvaluationTask
            - include_history: bool (or agent_type)
            - max_agent_turns: int
            - agent_type: Optional["context", "nocontext", "foundry"]
            - memory_store_name: Optional[str] (required if agent_type="foundry")
        on_result: Optional callback(index, context, result) called as each completes.

    Returns:
        List of MultiSessionEvaluationResult objects
    """
    pool = AsyncLLMPool()

    results = await pool.run_parallel(
        items=contexts,
        func=_run_single_evaluation_with_client,
        on_result=on_result,
    )

    return results
