"""
Evaluation Runner for Multi-Session Preference Recall.

Orchestrates the complete evaluation flow:
1. Load multi-session data
2. Generate evaluation task (via orchestrator)
3. Run dialogue between agent and user simulator (runner-managed state)
4. Evaluate with judge

The runner manages covered/uncovered state, selects the next preference to test
(deterministically from the planner's TESTING ORDER), and terminates the
conversation when all preferences are covered.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ream_bench.agents import ContextAwareAgent, NoContextAgent
from ream_bench.client import AsyncLLMPool, LLMClient, PooledLLMClient
from ream_bench.schemas import (
    EvaluationTaskSpec,
    MultiSessionEvaluationResult,
    MultiSessionOutput,
)
from ream_bench.task_generators import EvaluationTaskGenerator

from .judge import MultiSessionJudge
from .user_simulator import MultiSessionUserSimulator

# ---------------------------------------------------------------------------
# Runner state
# ---------------------------------------------------------------------------


@dataclass
class _RunnerState:
    """Deterministic state managed by the dialogue runner."""

    testing_order: list[str]
    covered: set[str] = field(default_factory=set)
    uncovered: set[str] = field(default_factory=set)
    verdicts: dict[str, str] = field(default_factory=dict)
    current_pref_id: str | None = None
    next_pref_id: str | None = None
    preferences: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_PREF_ID_RE = re.compile(r"pref_\d+")


def _parse_plan_testing_order(plan_text: str, fallback_ids: list[str]) -> list[str]:
    """Extract TESTING ORDER from the planner's output.

    Falls back to ``fallback_ids`` (rubric order) if parsing fails or if
    the parsed order doesn't contain all expected pref IDs.
    """
    match = re.search(r"TESTING ORDER:\s*\[([^\]]*)\]", plan_text)
    if not match:
        return fallback_ids

    ids = _PREF_ID_RE.findall(match.group(1))
    if not ids:
        return fallback_ids

    expected = set(fallback_ids)
    parsed = set(ids)

    # If parsed order is missing some prefs, append them at the end
    if parsed != expected:
        missing = [pid for pid in fallback_ids if pid not in parsed]
        ids = [pid for pid in ids if pid in expected] + missing

    return ids


def _extract_verdict(scratchpad_text: str) -> str | None:
    """Parse VERDICT field from v11 scratchpad text.

    Returns "recalled", "missed", or None.
    """
    match = re.search(r"VERDICT:\s*(\S+)", scratchpad_text, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower().strip().rstrip(".")
    if "recalled" in value:
        return "recalled"
    if "missed" in value:
        return "missed"
    if "n/a" in value:
        return None
    return None


def _extract_proactive_recalls(scratchpad_text: str, uncovered: set[str]) -> list[str]:
    """Parse PROACTIVE field from v11 scratchpad text.

    Returns list of pref IDs that the agent proactively addressed.
    Only returns IDs that are in the uncovered set.
    """
    match = re.search(r"PROACTIVE_RECALL:\s*(.+?)(?=\n|$)", scratchpad_text)
    if not match:
        return []
    value = match.group(1).strip()
    if value.lower() == "none":
        return []
    ids = _PREF_ID_RE.findall(value)
    return [pid for pid in ids if pid in uncovered]


def _next_uncovered_pref(state: _RunnerState) -> str | None:
    """Return the next uncovered pref from testing_order, or None if all covered."""
    for pid in state.testing_order:
        if pid in state.uncovered:
            return pid
    return None


def _parse_scratchpad(raw: str) -> dict[str, str | list[str]]:
    """Parse raw scratchpad text into a structured dict for JSON output.

    Supports both v10 fields (COVERED, UNCOVERED, EVALUATION, ACTION, TESTING)
    and v11 fields (VERDICT, PROACTIVE, REASONING).
    Falls back to returning the raw text if parsing fails.
    """
    result: dict[str, str | list[str]] = {}

    # --- v10 fields ---

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

    # --- v11 fields ---

    # VERDICT: RECALLED|MISSED|N/A
    verdict = re.search(r"VERDICT:\s*(\S+)", raw, re.IGNORECASE)
    if verdict:
        result["verdict"] = verdict.group(1).strip()

    # PROACTIVE_RECALL: [pref_ids] or none
    proactive = re.search(r"PROACTIVE_RECALL:\s*(.+?)(?=\n|$)", raw)
    if proactive:
        value = proactive.group(1).strip()
        if value.lower() == "none":
            result["proactive_recall"] = []
        else:
            result["proactive_recall"] = _PREF_ID_RE.findall(value)

    # REASONING: free-form text
    reasoning = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z]+:|\Z)", raw, flags=re.DOTALL)
    if reasoning:
        result["reasoning"] = reasoning.group(1).strip()

    # If we couldn't parse any fields, return raw text
    if not result:
        result["raw"] = raw

    return result


# ---------------------------------------------------------------------------
# Main evaluation entry points
# ---------------------------------------------------------------------------


def run_evaluation(
    multisession_data: MultiSessionOutput,
    max_agent_turns: int = 20,
    client: LLMClient | PooledLLMClient | None = None,
    eval_task: EvaluationTaskSpec | None = None,
    agent_type: Literal["context", "nocontext"] = "context",
    agent: Any = None,
    memory_token_budget: int | None = None,
) -> MultiSessionEvaluationResult:
    """Run a complete evaluation on multi-session data.

    Args:
        multisession_data: Output from MultiSessionGenerator
        max_agent_turns: Maximum agent turns before ending (default 20)
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
        memory_token_budget,
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
    memory_token_budget: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Run the evaluation dialogue between agent and user simulator.

    The runner manages covered/uncovered state and deterministically selects the
    next preference to test from the planner's TESTING ORDER. The user simulator
    LLM focuses on evaluating recall and writing natural messages.

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

    required_prefs = eval_task.rubric.required_preferences
    pref_lookup = {p["id"]: p for p in required_prefs}
    fallback_ids = [p["id"] for p in required_prefs]

    # Phase 1: Generate conversation plan
    print("  Generating conversation plan...", flush=True)
    plan_text = user_sim.generate_plan()

    # Phase 2: Initialize runner state
    testing_order = _parse_plan_testing_order(plan_text, fallback_ids)
    state = _RunnerState(
        testing_order=testing_order,
        uncovered=set(fallback_ids),
        preferences=pref_lookup,
    )
    state.next_pref_id = testing_order[0]

    # Phase 3: Generate opening message (probing for first pref)
    first_pref = pref_lookup[state.next_pref_id]
    user_message, initial_scratchpad = user_sim.generate_opening(plan_text, first_pref)

    conversation_with_scratchpads: list[dict[str, Any]] = []
    clean_conversation: list[dict[str, str]] = []

    conversation_with_scratchpads.append({
        "role": "user",
        "content": user_message,
        "scratchpad": initial_scratchpad,
        "plan": plan_text,
        "current_pref_id": None,
        "next_pref_id": state.next_pref_id,
    })
    clean_conversation.append({"role": "user", "content": user_message})

    # The pref we just probed for becomes the one to evaluate next turn
    state.current_pref_id = state.next_pref_id

    # Phase 4: Dialogue loop
    agent_turns = 0
    while agent_turns < max_agent_turns:
        # Agent responds
        agent_response, retrieved_memories = agent.respond(clean_conversation, memory_token_budget=memory_token_budget)
        print(
            f"  Turn {agent_turns + 1}: agent responded ({len(retrieved_memories)} memories retrieved)",
            flush=True,
        )
        conversation_with_scratchpads.append({
            "role": "assistant",
            "retrieved_memories": retrieved_memories if retrieved_memories else None,
            "content": agent_response,
        })
        clean_conversation.append({"role": "assistant", "content": agent_response})
        agent_turns += 1

        if agent_turns >= max_agent_turns:
            print(f"  Conversation hit max agent turns limit ({max_agent_turns})", flush=True)
            break

        # Determine next pref to probe for BEFORE user sim (needed for the prompt)
        # This is a preliminary pick — will be recomputed after proactive recalls
        state.next_pref_id = None
        for pid in state.testing_order:
            if pid in state.uncovered and pid != state.current_pref_id:
                state.next_pref_id = pid
                break

        # If nothing to evaluate and nothing to probe, we're done
        if state.current_pref_id is None and state.next_pref_id is None:
            break

        # User simulator: evaluate current pref, probe for next pref
        current_pref = pref_lookup.get(state.current_pref_id) if state.current_pref_id else None
        next_pref = pref_lookup.get(state.next_pref_id) if state.next_pref_id else None

        user_response, scratchpad = user_sim.respond(
            conversation_with_scratchpads,
            current_pref=current_pref,
            next_pref=next_pref,
            plan_text=plan_text,
        )

        # Extract verdict and update state
        if scratchpad and state.current_pref_id:
            verdict = _extract_verdict(scratchpad)
            if verdict:
                state.verdicts[state.current_pref_id] = verdict
            else:
                # Default to missed if we can't parse
                state.verdicts[state.current_pref_id] = "missed"
            state.covered.add(state.current_pref_id)
            state.uncovered.discard(state.current_pref_id)

            # Handle proactive recalls
            proactive = _extract_proactive_recalls(scratchpad, state.uncovered)
            for pid in proactive:
                state.verdicts[pid] = "recalled"
                state.covered.add(pid)
                state.uncovered.discard(pid)

        # Recompute next_pref_id AFTER proactive recalls to avoid re-testing
        # a preference that was just proactively recalled in this turn
        state.next_pref_id = None
        for pid in state.testing_order:
            if pid in state.uncovered and pid != state.current_pref_id:
                state.next_pref_id = pid
                break

        # Store turn with runner metadata
        conversation_with_scratchpads.append({
            "role": "user",
            "content": user_response,
            "scratchpad": scratchpad,
            "current_pref_id": state.current_pref_id,
            "next_pref_id": state.next_pref_id,
        })
        clean_conversation.append({"role": "user", "content": user_response})

        # Advance: what we just probed for becomes the next evaluation target
        state.current_pref_id = state.next_pref_id

        # Check if all preferences are covered — terminate
        if not state.uncovered:
            break

    return conversation_with_scratchpads, clean_conversation


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------


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
        memory_token_budget=context.get("memory_token_budget"),
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
