"""
Multi-Session User Simulator.

The user simulator acts as a realistic user during evaluation dialogues.
It generates its own opening message and drives the full conversation,
testing all required preferences through natural interaction.
"""

import re
from collections.abc import Mapping, Sequence

from memory_gym.client import CONFIG, ContentFilterError, LLMClient, PooledLLMClient
from memory_gym.prompts import render_prompt
from memory_gym.schemas import EvaluationTaskSpec


class MultiSessionUserSimulator:
    """Simulates a user in evaluation dialogues.

    The user simulator:
    - Generates its own opening message (no pre-generated prompt)
    - Acts naturally based on persona and preferences
    - Knows its current preferences (from required_preferences)
    - Corrects the agent when recommendations don't match preferences
    - Does NOT know which preferences are "stale" - it just knows what it wants now

    This creates realistic dialogue where the user's corrections emerge
    naturally from preference mismatches, not from explicit testing.
    """

    def __init__(
        self,
        evaluation_task: EvaluationTaskSpec,
        client: LLMClient | PooledLLMClient | None = None,
    ):
        """Initialize user simulator for a specific evaluation task.

        Args:
            evaluation_task: The evaluation task spec containing persona and preferences
            client: LLM client for generation. If None, creates a new one.
        """
        self.task = evaluation_task
        self.client = client or PooledLLMClient()

        # Extract preferences from rubric.required_preferences
        # These are dicts with {id, fact, supersedes?: ...}
        # User only needs to know their current preferences (ignores supersedes)
        self.required_preferences = evaluation_task.rubric.required_preferences

        # Build system prompt once
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the user simulator."""
        prefs_formatted = "\n".join(f"- {p['id']}: {p['fact']}" for p in self.required_preferences)

        return render_prompt(
            "user_simulator/user_simulator_system",
            persona_summary=self.task.persona,
            required_preferences=prefs_formatted,
        )

    def get_initial_message(self) -> tuple[str, str | None, str | None]:
        """Generate the initial user message to start the dialogue.

        Makes an LLM call using the system prompt plus an instruction to
        generate a natural opening message. Different each evaluation run.

        Returns:
            Tuple of (opening_message, scratchpad_or_none, plan_or_none)
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": "Generate your opening message to the AI assistant. Remember to include your scratchpad.",
            },
        ]

        response = self.client.complete_chat(
            messages=messages,
            max_tokens=CONFIG["max_tokens"]["user_simulator"],
        )

        raw_response = response.strip()
        clean_response, scratchpad = self._extract_scratchpad(raw_response)
        plan = self._extract_plan(raw_response)
        return clean_response, scratchpad, plan

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        conversation_with_scratchpads: list[dict[str, str | None]] | None = None,
    ) -> tuple[str, str | None]:
        """Generate user response to the latest agent message.

        Args:
            conversation_history: Clean conversation so far (without scratchpads)
            conversation_with_scratchpads: Conversation with scratchpad data on user turns.
                If provided, user simulator sees its own prior scratchpads for state continuity.

        Returns:
            Tuple of (user_response, scratchpad_or_none)
            - user_response: Clean response to add to conversation
            - scratchpad: Raw scratchpad content for debugging, or None if not present
        """
        history_for_prompt = conversation_with_scratchpads if conversation_with_scratchpads else conversation_history
        conv_text = self._format_conversation_as_string(history_for_prompt)
        user_prompt = render_prompt(
            "user_simulator/user_simulator_user",
            conversation=conv_text,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
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
            clean = re.sub(r"<plan>.*?</plan>\s*", "", clean, flags=re.DOTALL)
            return clean.strip(), scratchpad

        # Unclosed <scratchpad> tag means content filter truncated mid-generation
        if re.search(r"<scratchpad>", response):
            raise ContentFilterError("Content filter truncated response: unclosed <scratchpad> tag")

        clean = re.sub(r"<plan>.*?</plan>\s*", "", response, flags=re.DOTALL)
        return clean.strip(), None

    def _extract_plan(self, response: str) -> str | None:
        """Extract the <plan> block from a response.

        Returns:
            Plan content or None if no plan block found.
        """
        match = re.search(r"<plan>(.*?)</plan>", response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _format_conversation_as_string(self, history: Sequence[Mapping[str, str | None]]) -> str:
        """Format conversation history as labeled string for the prompt.

        If user turns contain a 'scratchpad' key, it is included so the model
        can see its own prior reasoning (COVERED/UNCOVERED state).
        The 'plan' key on the first user turn is also included.
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
