"""
Evaluation Runner for Multi-Session Preference Recall.

Orchestrates the complete evaluation flow:
1. Load multi-session data
2. Generate evaluation task (via orchestrator)
3. Run dialogue between agent and user simulator
4. Evaluate with judge
"""

import re
from typing import Any, Callable, Literal

from memory_gym.agents import (
    AWSMemoryAgent,
    ContextAwareAgent,
    FoundryLocalAgent,
    FoundryMemoryAgent,
    GoogleMemoryAgent,
    NoContextAgent,
)
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
    max_agent_turns: int = 10,
    client: LLMClient | PooledLLMClient | None = None,
    eval_task: EvaluationTaskSpec | None = None,
    agent_type: Literal["context", "nocontext", "foundry", "google", "aws", "foundry_local"] = "context",
    memory_store_name: str | None = None,
    foundry_agent: FoundryMemoryAgent | None = None,
    google_agent: GoogleMemoryAgent | None = None,
    aws_agent: AWSMemoryAgent | None = None,
    foundry_local_agent: FoundryLocalAgent | None = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    This function:
    1. Generates an evaluation task (or uses provided one)
    2. Runs a dialogue between the agent and user simulator
    3. Evaluates the dialogue with the judge

    Args:
        multisession_data: Output from MultiSessionGenerator
        max_agent_turns: Maximum agent turns before ending (default 10)
        client: Shared LLM client. If None, creates a new one.
        eval_task: Pre-generated evaluation task. If None, generates a new one.
        agent_type: Type of agent to use: "context", "nocontext", "foundry", "google", or "aws".
        memory_store_name: Name of memory store for foundry agent (required if agent_type="foundry").

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
        memory_store_name,
        foundry_agent,
        google_agent,
        aws_agent,
        foundry_local_agent,
    )

    # Step 3: Evaluate with judge (using clean conversation without scratchpads)
    judge = MultiSessionJudge(client)
    result = judge.evaluate(eval_task, clean_conversation)

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
    agent_type: Literal["context", "nocontext", "foundry", "google", "aws", "foundry_local"],
    client: LLMClient | PooledLLMClient,
    memory_store_name: str | None = None,
    foundry_agent: FoundryMemoryAgent | None = None,
    google_agent: GoogleMemoryAgent | None = None,
    aws_agent: AWSMemoryAgent | None = None,
    foundry_local_agent: FoundryLocalAgent | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run the evaluation dialogue between agent and user simulator.

    Args:
        eval_task: The evaluation task
        multisession_data: Source data (used to build agent context)
        max_agent_turns: Maximum agent turns
        agent_type: Type of agent: "context", "nocontext", "foundry", "google", or "aws"
        client: LLM client
        memory_store_name: Name of memory store for foundry agent

    Returns:
        Tuple of (conversation_with_scratchpads, clean_conversation)
        - conversation_with_scratchpads: List with "scratchpad" key on user turns (for output JSON)
        - clean_conversation: List without scratchpads (for judge)
    """
    user_sim = MultiSessionUserSimulator(eval_task, client)

    if agent_type == "foundry":
        if foundry_agent is not None:
            agent = foundry_agent
            agent.reset_conversation()
            agent.build_context(multisession_data)
        else:
            if not memory_store_name:
                raise ValueError("memory_store_name required for foundry agent")
            agent = FoundryMemoryAgent(memory_store_name=memory_store_name)
            agent.build_context(multisession_data)
    elif agent_type == "google":
        if google_agent is not None:
            agent = google_agent
            agent.reset_conversation()
            agent.build_context(multisession_data)
        else:
            agent = GoogleMemoryAgent()
            agent.build_context(multisession_data)
    elif agent_type == "aws":
        if aws_agent is not None:
            agent = aws_agent
            agent.reset_conversation()
            agent.build_context(multisession_data)
        else:
            if not memory_store_name:
                raise ValueError("memory_store_name required for aws agent")
            agent = AWSMemoryAgent(memory_name=memory_store_name)
            agent.build_context(multisession_data)
    elif agent_type == "foundry_local":
        if foundry_local_agent is not None:
            agent = foundry_local_agent
            agent.reset_conversation()
            agent.build_context(multisession_data)
        else:
            agent = FoundryLocalAgent()
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
    """Run a single evaluation using provided client (for parallel execution).

    Args:
        client: LLM client to use
        context: Dict containing multisession_data, eval_task, max_agent_turns,
                 and optionally agent_type, memory_store_name

    Returns:
        MultiSessionEvaluationResult
    """
    return run_evaluation(
        multisession_data=context["multisession_data"],
        eval_task=context["eval_task"],
        max_agent_turns=context["max_agent_turns"],
        client=client,
        agent_type=context.get("agent_type", "context"),
        memory_store_name=context.get("memory_store_name"),
        foundry_agent=context.get("foundry_agent"),
        google_agent=context.get("google_agent"),
        aws_agent=context.get("aws_agent"),
        foundry_local_agent=context.get("foundry_local_agent"),
    )


async def run_evaluations_parallel(
    contexts: list[dict],
    on_result: Callable[[int, dict, "MultiSessionEvaluationResult"], None] | None = None,
) -> list[MultiSessionEvaluationResult]:
    """Run multiple evaluations in parallel across deployments.

    Args:
        contexts: List of dicts, each containing:
            - multisession_data: MultiSessionOutput
            - eval_task: EvaluationTaskSpec
            - agent_type: "context", "nocontext", "foundry", "google", "aws", or "foundry_local" (default: "context")
            - max_agent_turns: int
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
