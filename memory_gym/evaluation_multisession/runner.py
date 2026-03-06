"""
Evaluation Runner for Multi-Session Preference Recall.

Orchestrates the complete evaluation flow:
1. Load multi-session data
2. Generate evaluation task (via orchestrator)
3. Run dialogue between agent and user simulator
4. Evaluate with judge
"""

import re
import time
from typing import Any, Callable, Literal

from memory_gym.agents import ContextAwareAgent, NoContextAgent
from memory_gym.client import AsyncLLMPool, LLMClient, PooledLLMClient
from memory_gym.schemas import (
    EvaluationTaskSpec,
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


def _parse_scratchpad(raw: str) -> dict[str, str | list[str]]:
    """Parse raw scratchpad text into a structured dict for JSON output.

    Extracts COVERED, UNCOVERED, EVALUATION, ACTION, and TESTING fields.
    Falls back to returning the raw text if parsing fails.
    """
    result: dict[str, str | list[str]] = {}

    # COVERED: [pref_004, pref_019] or COVERED: []
    covered = re.search(r"COVERED:\s*\[([^\]]*)\]", raw)
    if covered:
        items = [s.strip() for s in covered.group(1).split(",") if s.strip()]
        result["covered"] = items

    # UNCOVERED: [pref_052, pref_042] or UNCOVERED: []
    uncovered = re.search(r"UNCOVERED:\s*\[([^\]]*)\]", raw)
    if uncovered:
        items = [s.strip() for s in uncovered.group(1).split(",") if s.strip()]
        result["uncovered"] = items

    # EVALUATION: free-form text up to next field
    evaluation = re.search(r"EVALUATION:\s*(.+?)(?=\nACTION:|\nTESTING:|\Z)", raw, flags=re.DOTALL)
    if evaluation:
        result["evaluation"] = evaluation.group(1).strip()

    # ACTION: free-form text up to next field
    action = re.search(r"ACTION:\s*(.+?)(?=\nTESTING:|\Z)", raw, flags=re.DOTALL)
    if action:
        result["action"] = action.group(1).strip()

    # TESTING: pref_id — full fact text
    testing = re.search(r"TESTING:\s*(.+)", raw)
    if testing:
        result["testing"] = testing.group(1).strip()

    # If we couldn't parse any fields, return raw text
    if not result:
        result["raw"] = raw

    return result


def run_evaluation(
    multisession_data: MultiSessionOutput,
    max_agent_turns: int = 20,
    client: LLMClient | PooledLLMClient | None = None,
    eval_task: EvaluationTaskSpec | None = None,
    agent_type: Literal["context", "nocontext"] = "context",
    agent: Any = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    Args:
        multisession_data: Output from MultiSessionGenerator
        max_agent_turns: Maximum agent turns before ending (default 10)
        client: Shared LLM client. If None, creates a new one.
        eval_task: Pre-generated evaluation task. If None, generates a new one.
        agent_type: Baseline agent type: "context" or "nocontext". Ignored when ``agent`` is provided.
        agent: Pre-built agent (e.g. MemoryAgent). If provided, used directly.

    Returns:
        MultiSessionEvaluationResult with scores and analysis
    """
    client = client or PooledLLMClient()

    # Step 1: Generate or use provided evaluation task
    if eval_task is None:
        task_generator = EvaluationTaskGenerator()
        eval_task = task_generator.generate(multisession_data)

    # Step 2: Run dialogue
    conversation_with_scratchpads, clean_conversation = run_dialogue(
        eval_task,
        multisession_data,
        max_agent_turns,
        agent_type,
        client,
        agent,
    )

    # Step 3: Evaluate with judge (using clean conversation without scratchpads)
    # Pass scratchpad conversation so the judge can extract simulator verdicts
    judge = MultiSessionJudge(client)
    result = judge.evaluate(eval_task, clean_conversation, conversation_with_scratchpads)

    # Replace conversation with version that includes scratchpads for output
    # Parse raw scratchpad strings into structured dicts for readable JSON
    for turn in conversation_with_scratchpads:
        raw = turn.get("scratchpad")
        if isinstance(raw, str):
            turn["scratchpad"] = _parse_scratchpad(raw)  # type: ignore[assignment]
    result.conversation = conversation_with_scratchpads  # type: ignore[assignment]  # scratchpad adds structured data

    return result


def run_dialogue(
    eval_task: EvaluationTaskSpec,
    multisession_data: MultiSessionOutput,
    max_agent_turns: int,
    agent_type: Literal["context", "nocontext"],
    client: LLMClient | PooledLLMClient,
    agent: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run the evaluation dialogue between agent and user simulator.

    Args:
        eval_task: The evaluation task
        multisession_data: Source data (used to build agent context)
        max_agent_turns: Maximum agent turns
        agent_type: Baseline agent type (used when ``agent`` is None)
        client: LLM client
        agent: Pre-built agent. If provided, ``reset_conversation`` + ``build_context`` are called.

    Returns:
        Tuple of (conversation_with_scratchpads, clean_conversation)
    """
    user_sim = MultiSessionUserSimulator(eval_task, client)

    if agent is not None:
        agent.reset_conversation()
        agent.build_context(multisession_data)
    elif agent_type == "context":
        agent = ContextAwareAgent(client)
        agent.build_context(multisession_data)
    else:
        agent = NoContextAgent(client)
        agent.build_context(multisession_data)

    conversation_with_scratchpads: list[dict[str, Any]] = []
    clean_conversation: list[dict[str, str]] = []

    user_message, initial_scratchpad, plan = user_sim.get_initial_message()
    conversation_with_scratchpads.append(
        {"role": "user", "content": user_message, "scratchpad": initial_scratchpad, "plan": plan}
    )
    clean_conversation.append({"role": "user", "content": user_message})

    agent_turns = 0
    while agent_turns < max_agent_turns:
        agent_response, retrieved_memories = agent.respond(clean_conversation)
        print(f"  Turn {agent_turns + 1}: agent responded ({len(retrieved_memories)} memories retrieved)", flush=True)
        conversation_with_scratchpads.append(
            {
                "role": "assistant",
                "retrieved_memories": retrieved_memories if retrieved_memories else None,
                "content": agent_response,
            }
        )
        clean_conversation.append({"role": "assistant", "content": agent_response})
        agent_turns += 1

        if agent_turns >= max_agent_turns:
            print(f"Conversation hit max agent turns limit ({max_agent_turns})")
            break

        user_response, scratchpad = user_sim.respond(clean_conversation, conversation_with_scratchpads)
        conversation_with_scratchpads.append({"role": "user", "content": user_response, "scratchpad": scratchpad})
        clean_conversation.append({"role": "user", "content": user_response})

        if scratchpad and _is_uncovered_empty(scratchpad):
            break

    return conversation_with_scratchpads, clean_conversation


def _run_single_evaluation_with_client(
    client: LLMClient | PooledLLMClient,
    context: dict,
) -> MultiSessionEvaluationResult:
    """Run a single evaluation using provided client (for parallel execution)."""
    t0 = time.monotonic()
    result = run_evaluation(
        multisession_data=context["multisession_data"],
        eval_task=context["eval_task"],
        max_agent_turns=context["max_agent_turns"],
        client=client,
        agent_type=context.get("agent_type", "context"),
        agent=context.get("agent"),
    )
    result.eval_seconds = round(time.monotonic() - t0, 2)
    return result


async def run_evaluations_parallel(
    contexts: list[dict],
    on_result: Callable[[int, dict, "MultiSessionEvaluationResult"], None] | None = None,
) -> list[MultiSessionEvaluationResult]:
    """Run multiple evaluations in parallel across deployments.

    Args:
        contexts: List of dicts, each containing:
            - multisession_data: MultiSessionOutput
            - eval_task: EvaluationTaskSpec
            - agent_type: "context" or "nocontext" (default: "context"). Ignored when ``agent`` is set.
            - max_agent_turns: int
            - agent: Optional pre-built agent (MemoryAgent or baseline)
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
