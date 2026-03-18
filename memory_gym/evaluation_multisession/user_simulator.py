"""
Multi-Session User Simulator.

The user simulator acts as a realistic user during evaluation dialogues.
It generates a conversation plan (via a dedicated planner LLM call), then
drives evaluation turns where it evaluates recall and writes probing messages.
State management (covered/uncovered, next-preference selection, termination)
is handled by the dialogue runner — the simulator focuses on evaluation and
natural message writing.
"""

import re
from collections.abc import Mapping, Sequence

from memory_gym.client import CONFIG, ContentFilterError, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import EvaluationTaskSpec


class MultiSessionUserSimulator:
    """Simulates a user in evaluation dialogues.

    Three-phase interface driven by the dialogue runner:
    1. generate_plan() — one-shot planner LLM call (topic groups, testing order, bridges)
    2. generate_opening() — first user message probing for the first preference
    3. respond() — per-turn: evaluate current pref, write message probing for next pref
    """

    def __init__(
        self,
        evaluation_task: EvaluationTaskSpec,
        client: LLMClient | PooledLLMClient | None = None,
    ):
        self.task = evaluation_task
        self.client = client or PooledLLMClient()
        self.required_preferences = evaluation_task.rubric.required_preferences
        self._persona_summary = evaluation_task.persona

        # Formatted preference list for the planner
        self._prefs_formatted = "\n".join(f"- {p['id']}: {p['fact']}" for p in self.required_preferences)

    def generate_plan(self) -> str:
        """Generate a conversation plan via the planner LLM.

        Returns:
            Raw plan text (TOPIC GROUPS, TESTING ORDER, BRIDGES, PROBE STRATEGIES).
        """
        system_prompt = render_prompt(
            "user_simulator/user_simulator_plan_system",
            persona_summary=self._persona_summary,
            required_preferences=self._prefs_formatted,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Generate the conversation plan now. Output ONLY the plan in the specified format.",
            },
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["user_simulator"],
        )

        return response.strip()

    def generate_opening(
        self,
        plan_text: str,
        first_pref: dict,
    ) -> tuple[str, str | None]:
        """Generate the opening user message.

        Args:
            plan_text: The conversation plan from generate_plan()
            first_pref: The first preference to probe for (dict with id, fact)

        Returns:
            Tuple of (opening_message, scratchpad_or_none)
        """
        system_prompt = render_prompt(
            "user_simulator/user_simulator_system",
            persona_summary=self._persona_summary,
            current_preference="N/A (first turn — no agent reply yet)",
            next_preference=f"{first_pref['id']}: {first_pref['fact']}",
            remaining_count=len(self.required_preferences),
            plan=plan_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Generate your opening message to the AI assistant. "
                    "This is your first message — there is no agent reply to evaluate yet. "
                    "Write a natural opening that probes for the preference indicated. "
                    "Remember to include your scratchpad."
                ),
            },
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["user_simulator"],
        )

        raw_response = response.strip()
        clean_response, scratchpad = self._extract_scratchpad(raw_response)
        return clean_response, scratchpad

    def respond(
        self,
        conversation_with_scratchpads: list[dict],
        current_pref: dict | None,
        next_pref: dict | None,
        plan_text: str,
    ) -> tuple[str, str | None]:
        """Generate a user response for the current turn.

        Args:
            conversation_with_scratchpads: Full conversation with scratchpad data on user turns.
            current_pref: The preference to EVALUATE (tested last turn). None if nothing to evaluate.
            next_pref: The preference to PROBE FOR next. None if this is the last evaluation turn.
            plan_text: The conversation plan.

        Returns:
            Tuple of (user_response, scratchpad_or_none)
        """
        current_pref_text = (
            f"{current_pref['id']}: {current_pref['fact']}" if current_pref else "N/A"
        )
        next_pref_text = (
            f"{next_pref['id']}: {next_pref['fact']}" if next_pref else "none"
        )

        system_prompt = render_prompt(
            "user_simulator/user_simulator_system",
            persona_summary=self._persona_summary,
            current_preference=current_pref_text,
            next_preference=next_pref_text,
            plan=plan_text,
        )

        conv_text = self._format_conversation_as_string(conversation_with_scratchpads)
        user_prompt = render_prompt(
            "user_simulator/user_simulator_user",
            conversation=conv_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["user_simulator"],
        )

        raw_response = response.strip()
        clean_response, scratchpad = self._extract_scratchpad(raw_response)
        return clean_response, scratchpad

    def _extract_scratchpad(self, response: str) -> tuple[str, str | None]:
        """Extract scratchpad and clean response separately.

        Raises ContentFilterError if an opening <scratchpad> tag is found
        without a matching closing tag — this indicates the content filter
        truncated the response mid-generation.

        Returns:
            Tuple of (clean_response, scratchpad_content_or_none)
        """
        match = re.search(r"<scratchpad>(.*?)</scratchpad>", response, flags=re.DOTALL)
        if match:
            scratchpad = match.group(1).strip()
            clean = re.sub(r"<scratchpad>.*?</scratchpad>\s*", "", response, flags=re.DOTALL)
            return clean.strip(), scratchpad

        # Unclosed <scratchpad> tag means content filter truncated mid-generation
        if re.search(r"<scratchpad>", response):
            raise ContentFilterError("Content filter truncated response: unclosed <scratchpad> tag")

        return response.strip(), None

    def _format_conversation_as_string(self, history: Sequence[Mapping[str, str | None]]) -> str:
        """Format conversation history as labeled string for the prompt.

        If user turns contain a 'scratchpad' key, it is included so the model
        can see its own prior reasoning (verdicts). The 'plan' key on the first
        user turn is also included.
        """
        lines = []
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Agent"
            content = turn.get("content", "")
            plan = turn.get("plan")
            scratchpad = turn.get("scratchpad")
            if role == "User" and (plan or scratchpad):
                parts = []
                if plan:
                    parts.append(f"<plan>\n{plan}\n</plan>")
                if scratchpad:
                    parts.append(f"<scratchpad>\n{scratchpad}\n</scratchpad>")
                parts.append(str(content))
                lines.append(f"{role}:\n" + "\n\n".join(parts))
            else:
                lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
