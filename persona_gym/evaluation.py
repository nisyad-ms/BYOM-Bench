#!/usr/bin/env python3
"""
evaluation.py - Run and Evaluate Task-Oriented Dialogues

This module orchestrates TOD evaluation:
1. Loads conversation history (user preferences/context)
2. Runs multi-turn dialogue with agent under evaluation
3. Uses simulated user to provide task and respond
4. Evaluates agent performance using LLM judge

Usage:
    python -m persona_gym.evaluation --context <shared_context.json> --task <tod_task.json> --agent openai
"""

# =============================================================================
# PATH SETUP - Must come before ANY imports from this project
# =============================================================================
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from persona_gym.agent import BaseAgent, create_agent
from persona_gym.metric import (
    PreferenceItem,
    PreferenceUsage,
    TODEvaluationResult,
    TODTask,
    TurnAnalysis,
    TurnType,
    build_judge_prompt,
    parse_judge_response,
)
from persona_gym.tool_simulator import ToolResponse, create_simulator_from_task

load_dotenv()

# Log to project root logs/ folder (one level up from package)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'evaluation.log'))
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Simulated User (for evaluation)
# =============================================================================

class SimulatedUser:
    """Simulated user that provides tasks and responds to agent."""

    def __init__(
        self,
        task: TODTask,
        deployment: Optional[str] = None,
    ):
        self.task = task

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview"
        )
        self.deployment = deployment
        self.turn_count = 0
        self.max_turns = 15
        self.task_completed = False

    def get_initial_message(self) -> str:
        """Get the initial task message from the user."""
        return self.task.task_description

    def respond(self, agent_message: str, tool_results: Optional[List[ToolResponse]] = None) -> str:
        """Generate user response to agent's message."""
        self.turn_count += 1

        # Check for task completion signals in agent's message
        # Only mark complete if agent explicitly confirms a booking/reservation
        completion_keywords = ["booked", "confirmed", "reservation complete", "booking confirmed", "all set"]
        agent_lower = agent_message.lower()
        if any(kw in agent_lower for kw in completion_keywords):
            # Verify it's a confirmation, not a question about booking
            if "?" not in agent_message[-50:]:  # No question at the end
                self.task_completed = True
                return "Great, thank you!"

        # Check for max turns
        if self.turn_count >= self.max_turns:
            self.task_completed = True
            return "I need to go now. Let's wrap this up."

        # Generate contextual response
        preferences_str = "\n".join([f"- {p.fact}" for p in self.task.relevant_preferences])

        prompt = f"""Continue this {self.task.topic} assistant conversation as the customer.

Customer's task: {self.task.task_description}
Customer's preferences (internal, don't volunteer): {preferences_str}

Assistant said: "{agent_message}"

{f"Options shown: {json.dumps(tool_results)}" if tool_results else ""}

Write the customer's next message. Rules:
- Only answer what was asked, don't add extra info
- If asked a preference question, answer honestly
- If shown options, pick one matching preferences or reject bad ones
- Keep it to 1-2 sentences
- Don't explain preferences unless asked why

Customer:"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )

        user_response = response.choices[0].message.content.strip()

        # Only mark complete if user response is a clear thank-you for completion
        # AND the agent's previous message indicated completion
        if self.turn_count >= 4:  # Minimum turns before completion is reasonable
            if user_response.lower().strip() in ["great, thank you!", "great, thanks!", "thank you!", "thanks!"]:
                self.task_completed = True

        return user_response


# =============================================================================
# Dialogue Runner
# =============================================================================

@dataclass
class DialogueTurn:
    """A single turn in the dialogue."""
    turn_number: int
    speaker: str  # "user" or "agent"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None


def run_tod_dialogue(
    agent: BaseAgent,
    task: TODTask,
    context_messages: Optional[List[Dict]] = None,
    max_turns: int = 20,
    verbose: bool = False,
) -> List[DialogueTurn]:
    """
    Run a complete TOD dialogue between agent and simulated user.

    Args:
        agent: The agent being evaluated
        task: The TOD task to complete
        context_messages: Historical conversation context
        max_turns: Maximum dialogue turns
        verbose: Print dialogue as it happens

    Returns:
        List of DialogueTurn objects
    """
    dialogue = []
    conversation_history = []

    # Initialize simulated user and tool simulator
    user = SimulatedUser(task)
    tool_simulator = create_simulator_from_task(task)

    # User initiates with task
    user_message = user.get_initial_message()
    dialogue.append(DialogueTurn(
        turn_number=1,
        speaker="user",
        content=user_message,
    ))
    conversation_history.append({"role": "user", "content": user_message})

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.task_description}")
        print(f"{'='*60}")
        print(f"\n[User]: {user_message}")

    turn_num = 2
    while turn_num <= max_turns and not user.task_completed:
        # Agent responds
        agent_response = agent.respond(
            conversation_history=conversation_history,
            user_message=user_message,
            available_tools=task.tool_schemas,
            context_messages=context_messages,
        )

        agent_content = agent_response["content"]
        tool_calls = agent_response.get("tool_calls", [])

        # Process tool calls
        tool_results = []
        if tool_calls:
            for tc in tool_calls:
                result = tool_simulator.call_tool(tc["name"], tc.get("params", {}))
                tool_results.append(result.to_agent_view())

                # Append tool result to agent's response
                agent_content += f"\n\n[Tool Result from {tc['name']}]: {json.dumps(result.to_agent_view()['results'][:3], indent=2)}"

        dialogue.append(DialogueTurn(
            turn_number=turn_num,
            speaker="agent",
            content=agent_content,
            tool_calls=tool_calls,
            tool_results=tool_results,
        ))
        conversation_history.append({"role": "assistant", "content": agent_content})

        if verbose:
            print(f"\n[Agent]: {agent_content[:500]}{'...' if len(agent_content) > 500 else ''}")
            if tool_calls:
                print(f"  [Tools called: {[tc['name'] for tc in tool_calls]}]")

        turn_num += 1

        # Check if task seems complete
        if user.task_completed:
            break

        # User responds
        user_message = user.respond(agent_content, tool_results)
        dialogue.append(DialogueTurn(
            turn_number=turn_num,
            speaker="user",
            content=user_message,
        ))
        conversation_history.append({"role": "user", "content": user_message})

        if verbose:
            print(f"\n[User]: {user_message}")

        turn_num += 1

    return dialogue


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_dialogue(
    dialogue: List[DialogueTurn],
    task: TODTask,
    judge_deployment: Optional[str] = None,
) -> TODEvaluationResult:
    """
    Evaluate a completed dialogue using LLM judge.

    Args:
        dialogue: The completed dialogue turns
        task: The TOD task that was attempted
        judge_deployment: Azure OpenAI deployment for judge

    Returns:
        TODEvaluationResult with scores and analysis
    """
    # Format dialogue as transcript for judge
    transcript = []
    for turn in dialogue:
        transcript.append({
            "speaker": turn.speaker,
            "content": turn.content,
        })
        if turn.tool_calls:
            transcript.append({
                "speaker": "agent",
                "tool_call": turn.tool_calls[0]["name"],
                "params": turn.tool_calls[0].get("params", {}),
            })

    # Build judge prompt
    system_prompt, user_prompt = build_judge_prompt(task, transcript)

    # Call judge
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = judge_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-12-01-preview"
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.0,
    )

    judge_response = response.choices[0].message.content

    # Parse judge response
    try:
        eval_data = parse_judge_response(judge_response)
    except Exception as e:
        logger.error(f"Failed to parse judge response: {e}")
        logger.debug(f"Judge response: {judge_response}")
        # Return a default result
        return TODEvaluationResult(
            task_id=task.task_id,
            task_completed=False,
            turn_classifications=[],
            preference_usage={},
            total_turns=len(dialogue),
            correction_turns=0,
            repeated_correction_turns=0,
            efficiency_score=0.0,
            preference_score=0.0,
            task_success_score=0.0,
            final_score=0.0,
            reasoning=f"Judge parse error: {e}",
        )

    # Convert to result object
    turn_analyses = [
        TurnAnalysis(
            turn_number=t.get("turn", 0),
            speaker=t.get("speaker", "unknown"),
            content="",
            turn_type=TurnType(t.get("type", "productive")),
            reasoning=t.get("reasoning", ""),
        )
        for t in eval_data.get("turn_classifications", [])
    ]

    preference_usage = {
        pref: PreferenceUsage(usage)
        for pref, usage in eval_data.get("preference_usage", {}).items()
    }

    # Compute scores ourselves to ensure correctness
    task_completed = eval_data.get("task_completed", False)
    task_success_score = 1.0 if task_completed else 0.0

    correction_turns = eval_data.get("correction_turns", 0)
    repeated_correction_turns = eval_data.get("repeated_correction_turns", 0)

    # Efficiency score
    if repeated_correction_turns > 0:
        efficiency_score = 0.0
    else:
        efficiency_score = max(0.0, 1.0 - (correction_turns * 0.25))

    # Preference score: count proactive / (proactive + ignored)
    proactive_count = sum(1 for usage in preference_usage.values() if usage == PreferenceUsage.PROACTIVE)
    ignored_count = sum(1 for usage in preference_usage.values() if usage == PreferenceUsage.IGNORED)
    applicable_count = proactive_count + ignored_count
    preference_score = proactive_count / applicable_count if applicable_count > 0 else 1.0

    # Final score
    final_score = 0.34 * task_success_score + 0.33 * preference_score + 0.33 * efficiency_score

    return TODEvaluationResult(
        task_id=task.task_id,
        task_completed=task_completed,
        turn_classifications=turn_analyses,
        preference_usage=preference_usage,
        total_turns=eval_data.get("total_turns", len(dialogue)),
        correction_turns=correction_turns,
        repeated_correction_turns=repeated_correction_turns,
        efficiency_score=efficiency_score,
        preference_score=preference_score,
        task_success_score=task_success_score,
        final_score=final_score,
        reasoning=eval_data.get("reasoning", ""),
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def run_full_evaluation(
    context_path: str,
    task_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
    no_context: bool = False,
    agent_type: str = "context",
) -> Dict[str, Any]:
    """
    Run full TOD evaluation pipeline.

    Args:
        context_path: Path to shared context JSON
        task_path: Path to TOD task JSONL
        output_path: Path to save results
        verbose: Print progress
        no_context: If True, agent receives only the task query (no conversation history)
        agent_type: Type of agent to use ('context', 'no_context')

    Returns:
        Dict with evaluation results
    """
    # Load context
    logger.info(f"Loading context from {context_path}")
    with open(context_path, 'r') as f:
        context_data = json.load(f)

    # Extract messages from context (handle different formats)
    if no_context:
        # No memory baseline - agent only receives task query
        context_messages = []
        logger.info("Running in NO-CONTEXT mode (baseline without memory)")
    elif isinstance(context_data, list):
        # New format: direct list of {role, content} messages
        context_messages = context_data
        logger.info(f"Loaded {len(context_messages)} context messages (new format)")
    elif isinstance(context_data, dict):
        # Legacy format: dict with 'conversation' key
        if 'conversation' in context_data:
            context_messages = context_data['conversation']
            logger.info(f"Loaded {len(context_messages)} context messages (legacy format)")
        else:
            # Could be {context_id: messages} format
            context_messages = list(context_data.values())[0] if context_data else []
    else:
        context_messages = []
        logger.warning(f"Unknown context format: {type(context_data)}")

    # Load tasks
    logger.info(f"Loading tasks from {task_path}")
    tasks = []
    with open(task_path, 'r') as f:
        for line in f:
            if line.strip():
                task_data = json.loads(line)
                # Convert 'type' key to 'preference_type' for PreferenceItem
                prefs = []
                for p in task_data["relevant_preferences"]:
                    prefs.append(PreferenceItem(
                        fact=p.get("fact", ""),
                        preference_type=p.get("type", p.get("preference_type", "current")),
                        source_date=p.get("source_date", ""),
                        old_value=p.get("old_value"),
                        reason_of_change=p.get("reason_of_change"),
                    ))
                tasks.append(TODTask(
                    task_id=task_data["task_id"],
                    task_description=task_data["task_description"],
                    topic=task_data["topic"],
                    relevant_preferences=prefs,
                    expected_behaviors=task_data["expected_behaviors"],
                    tool_schemas=task_data.get("tool_schemas", {}),
                ))

    logger.info(f"Loaded {len(tasks)} tasks")

    # Initialize agent based on type
    # Note: if no_context flag is set, override to use no_context agent
    effective_agent_type = "no_context" if no_context else agent_type
    agent = create_agent(effective_agent_type)
    logger.info(f"Using agent: {agent.name} (uses_context={agent.uses_context})")

    # Run evaluations
    results = []
    for i, task in enumerate(tasks):
        logger.info(f"Running task {i+1}/{len(tasks)}: {task.task_description[:50]}...")

        try:
            # Run dialogue
            dialogue = run_tod_dialogue(
                agent=agent,
                task=task,
                context_messages=context_messages,
                verbose=verbose,
            )

            # Evaluate
            eval_result = evaluate_dialogue(dialogue, task)

            results.append({
                "task": task.to_dict(),
                "dialogue": [asdict(turn) for turn in dialogue],
                "evaluation": eval_result.to_dict(),
            })

            logger.info(f"  Score: {eval_result.final_score:.2f}")
        except Exception as e:
            logger.warning(f"  Task failed with error: {e}")
            logger.warning("  Skipping task and continuing...")
            continue

    # Compute aggregate scores
    if results:
        avg_score = sum(r["evaluation"]["scores"]["final_score"] for r in results) / len(results)
        avg_preference = sum(r["evaluation"]["scores"]["preference_score"] for r in results) / len(results)
        avg_efficiency = sum(r["evaluation"]["scores"]["efficiency_score"] for r in results) / len(results)
    else:
        avg_score = avg_preference = avg_efficiency = 0.0

    output = {
        "timestamp": datetime.now().isoformat(),
        "context_path": context_path,
        "task_path": task_path,
        "num_tasks": len(tasks),
        "agent_type": agent.name,
        "agent_uses_context": agent.uses_context,
        "evaluation_mode": "no_context" if no_context else "with_context",
        "aggregate_scores": {
            "average_final_score": avg_score,
            "average_preference_score": avg_preference,
            "average_efficiency_score": avg_efficiency,
        },
        "results": results,
    }

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run TOD evaluation")
    parser.add_argument('--context', '-c', type=str, required=True,
                        help='Path to shared context JSON')
    parser.add_argument('--tasks', '-t', type=str, required=True,
                        help='Path to TOD tasks JSONL')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--agent', '-a', type=str, default='context',
                        choices=['context', 'no_context'],
                        help='Agent type: context (with memory), no_context (baseline)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print dialogue progress')
    parser.add_argument('--no-context', action='store_true',
                        help='Run without conversation context (baseline without memory)')

    args = parser.parse_args()

    # Handle agent type: --no-context flag overrides --agent
    agent_type = 'no_context' if args.no_context else args.agent

    if args.output is None:
        base = os.path.splitext(args.tasks)[0]
        suffix = f"_{agent_type}" if agent_type != "context" else ""
        args.output = f"{base}_evaluation_results{suffix}.json"

    results = run_full_evaluation(
        context_path=args.context,
        task_path=args.tasks,
        output_path=args.output,
        verbose=args.verbose,
        no_context=args.no_context,
        agent_type=agent_type,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("TOD Evaluation Summary")
    print(f"{'='*60}")
    print(f"Agent: {results.get('agent_type', 'unknown')}")
    print(f"Tasks evaluated: {results['num_tasks']}")
    print(f"Evaluation mode: {results['evaluation_mode']}")
    print("\nAggregate Scores:")
    print(f"  Final Score:      {results['aggregate_scores']['average_final_score']:.2f}")
    print(f"  Preference Score: {results['aggregate_scores']['average_preference_score']:.2f}")
    print(f"  Efficiency Score: {results['aggregate_scores']['average_efficiency_score']:.2f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
