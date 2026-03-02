"""
Render all active prompts (from configs/prompts.yaml) with real variable values
and save the rendered output to prompt_renders/ at the repo root.

Usage:
    uv run python render_prompts.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
PROMPT_RENDERS_DIR = REPO_ROOT / "prompt_renders"


# ---------------------------------------------------------------------------
# Helper: find first timestamped output directory
# ---------------------------------------------------------------------------
def find_first_output_dir() -> Path | None:
    """Return the first timestamped subdirectory in outputs/, sorted alphabetically."""
    if not OUTPUTS_DIR.exists():
        return None
    candidates = sorted(
        p for p in OUTPUTS_DIR.iterdir() if p.is_dir()
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Helper: load sessions.json from an output directory
# ---------------------------------------------------------------------------
def load_sessions(output_dir: Path) -> list[dict]:
    sessions_path = output_dir / "sessions.json"
    if not sessions_path.exists():
        print(f"ERROR: sessions.json not found in {output_dir}/")
        sys.exit(1)
    with open(sessions_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: build an ExpandedPersona-compatible object from raw JSON data.
# The stored JSON may use legacy field names (work_and_education, etc.)
# so we normalise here rather than rely on ExpandedPersona.from_dict().
# ---------------------------------------------------------------------------
def build_expanded_persona(ep_data: dict):
    """Return a memory_gym.schemas.ExpandedPersona from raw dict."""
    from memory_gym.schemas import ExpandedPersona  # noqa: PLC0415

    # Normalise field names: the prompt generation code uses the schema's
    # canonical names (work_education, health_wellness, family_relationships),
    # but older saved data may use human-readable keys.
    field_aliases = {
        "work_education": ["work_education", "work_and_education"],
        "health_wellness": ["health_wellness", "health_and_wellness"],
        "family_relationships": ["family_relationships", "relationships_and_personal"],
    }

    resolved: dict = dict(ep_data)  # shallow copy
    for canonical, aliases in field_aliases.items():
        if canonical not in resolved or not resolved[canonical]:
            for alias in aliases:
                if alias in ep_data and ep_data[alias]:
                    resolved[canonical] = ep_data[alias]
                    break

    return ExpandedPersona.from_dict(resolved)


# ---------------------------------------------------------------------------
# Helper: convert baseline_preferences dict to flat JSON list with synth IDs
# ---------------------------------------------------------------------------
def baseline_prefs_to_json(baseline_preferences: dict[str, list[str]]) -> str:
    items = []
    counter = 1
    for domain, prefs in baseline_preferences.items():
        for fact in prefs:
            items.append(
                {
                    "preference_id": f"pref_{counter:03d}",
                    "fact": fact,
                    "domain": domain,
                }
            )
            counter += 1
    return json.dumps(items, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helper: save rendered text to the correct path under prompt_renders/
# ---------------------------------------------------------------------------
def save_render(prompt_name: str, rendered: str) -> Path:
    out_path = PROMPT_RENDERS_DIR / f"{prompt_name}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Helper: build a MultiSessionOutput from raw persona_data dict
# ---------------------------------------------------------------------------
def build_multisession_output(persona_data: dict):
    """Deserialize a MultiSessionOutput from raw sessions.json persona dict."""
    from memory_gym.schemas import MultiSessionOutput  # noqa: PLC0415

    return MultiSessionOutput.from_dict(persona_data)


# ---------------------------------------------------------------------------
# Helper: load first task from tasks directory, fail hard if missing
# ---------------------------------------------------------------------------
def load_first_task(output_dir: Path) -> dict:
    """Load the first available task JSON from tasks/<version>/. Exits if none found."""
    from utils import get_all_tasks, get_latest_task_version  # noqa: PLC0415

    version = get_latest_task_version(output_dir)
    if version is None:
        print(f"ERROR: tasks not found in {output_dir.name}/ — run test_task_generation.py first")
        sys.exit(1)

    task_paths = get_all_tasks(output_dir, version)
    if not task_paths:
        print(f"ERROR: tasks not found in {output_dir.name}/tasks/{version}/ — run test_task_generation.py first")
        sys.exit(1)

    with open(task_paths[0], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: load first eval result, fail hard if missing
# ---------------------------------------------------------------------------
def load_first_eval_result(output_dir: Path) -> dict | None:
    """Load the first available eval result JSON from evaluations/<run>/. Returns None if not found."""
    from utils import get_latest_eval_run_dir  # noqa: PLC0415

    eval_run_dir = get_latest_eval_run_dir(output_dir)
    if eval_run_dir is None:
        return None

    result_paths = sorted(eval_run_dir.glob("eval_*.json"))
    if not result_paths:
        return None

    with open(result_paths[0], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # -- Locate output data --------------------------------------------------
    output_dir = find_first_output_dir()
    if output_dir is None:
        print("ERROR: No output directories found in outputs/ — run test_data_generation.py first")
        sys.exit(1)

    print(f"Using output directory: {output_dir.relative_to(REPO_ROOT)}")

    # Load top-level sessions data (the first persona in the file)
    data = load_sessions(output_dir)

    # sessions.json stores a single MultiSessionOutput dict (not a list)
    if isinstance(data, list):
        persona_data = data[0]
    else:
        persona_data = data

    base_persona: str = persona_data["persona"]

    # Build ExpandedPersona
    ep_data = persona_data.get("expanded_persona", {})
    expanded_persona = build_expanded_persona(ep_data)

    # Raw sessions list
    raw_sessions: list[dict] = persona_data.get("sessions", [])
    session0 = raw_sessions[0] if raw_sessions else {}

    # Build MultiSessionOutput for format_preference_history
    multisession_output = build_multisession_output(persona_data)

    # Load real task data — fail hard if tasks don't exist
    task_data = load_first_task(output_dir)
    from memory_gym.schemas import EvaluationTaskSpec  # noqa: PLC0415

    task_spec = EvaluationTaskSpec.from_dict(task_data)
    required_preferences = task_spec.rubric.required_preferences

    # -- Import render_prompt from the package --------------------------------
    import yaml  # noqa: PLC0415

    from memory_gym.formatting import format_preference_history  # noqa: PLC0415
    from memory_gym.prompts import render_prompt  # noqa: PLC0415

    config_path = REPO_ROOT / "configs" / "prompts.yaml"
    with open(config_path, encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f) or {}
    active_prompts: dict[str, str] = prompt_config.get("prompts", {})

    print(f"Found {len(active_prompts)} active prompts in configs/prompts.yaml\n")

    # -- Render each prompt ---------------------------------------------------
    for prompt_name in active_prompts:
        # ------------------------------------------------------------------
        # data_generation/multisession/expand_life_facts_system
        # No variables — render as-is.
        # ------------------------------------------------------------------
        if prompt_name == "data_generation/multisession/expand_life_facts_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # data_generation/multisession/expand_life_facts_user
        # Variable: persona = base_persona (1-line string from sessions.json)
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/expand_life_facts_user":
            rendered = render_prompt(prompt_name, persona=base_persona)

        # ------------------------------------------------------------------
        # data_generation/multisession/generate_baseline_preferences_system
        # No variables.
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/generate_baseline_preferences_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # data_generation/multisession/generate_baseline_preferences_user
        # Variable: persona = life circumstances only (no preferences yet)
        # Mirror what the generator does: clear baseline_preferences before
        # calling to_full_description() so preferences aren't leaked into the
        # input of the prompt that's supposed to generate them.
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/generate_baseline_preferences_user":
            from dataclasses import replace  # noqa: PLC0415

            persona_no_prefs = replace(expanded_persona, baseline_preferences={})
            rendered = render_prompt(
                prompt_name,
                persona=persona_no_prefs.to_full_description(),
            )

        # ------------------------------------------------------------------
        # data_generation/multisession/generate_life_event_system
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/generate_life_event_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # data_generation/multisession/generate_life_event_user
        # Variables:
        #   persona  = expanded_persona.to_full_description()
        #   domain   = sessions[0].life_event.domain
        #   previous_events = "None (this is the first event)" for session 0
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/generate_life_event_user":
            domain = session0.get("life_event", {}).get("domain", "work_habits")
            rendered = render_prompt(
                prompt_name,
                persona=expanded_persona.to_full_description(),
                domain=domain,
                previous_events="None (this is the first event)",
            )

        # ------------------------------------------------------------------
        # data_generation/multisession/update_preferences_system
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/update_preferences_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # data_generation/multisession/update_preferences_user
        # Variables:
        #   persona            = expanded_persona.to_full_description()
        #   current_event      = sessions[0].life_event.event
        #   event_date         = sessions[0].life_event.date
        #   previous_events    = "None" for session 0
        #   active_preferences = baseline prefs formatted as JSON list
        #   evolution_history  = "None" for session 0
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/update_preferences_user":
            life_event0 = session0.get("life_event", {})
            active_preferences_json = baseline_prefs_to_json(
                expanded_persona.baseline_preferences or {}
            )
            rendered = render_prompt(
                prompt_name,
                persona=expanded_persona.to_full_description(),
                current_event=life_event0.get("event", ""),
                event_date=life_event0.get("date", ""),
                previous_events="None",
                active_preferences=active_preferences_json,
                evolution_history="None",
            )

        # ------------------------------------------------------------------
        # data_generation/multisession/generate_session_conversation_user
        # Variables:
        #   persona       = expanded_persona.to_full_description()
        #   life_event    = sessions[0].life_event.event
        #   event_date    = sessions[0].life_event.date
        #   session_delta = reconstruct from sessions[0].preferences (created + evolved)
        #   session_id    = 0
        # ------------------------------------------------------------------
        elif prompt_name == "data_generation/multisession/generate_session_conversation_user":
            life_event0 = session0.get("life_event", {})
            prefs_data = session0.get("preferences", {})

            session_delta = []

            # Evolved preferences
            for evo in prefs_data.get("evolved", []):
                session_delta.append(
                    {
                        "type": "evolved",
                        "old": evo["from"]["fact"],
                        "new": evo["to"]["fact"],
                        "reason": evo.get("reason", ""),
                    }
                )

            # Newly created preferences (exclude those that are the "to" side of evolutions)
            evolved_new_ids = {evo["to"]["id"] for evo in prefs_data.get("evolved", [])}
            for created in prefs_data.get("created", []):
                if created["id"] not in evolved_new_ids:
                    session_delta.append(
                        {
                            "type": "new",
                            "fact": created["fact"],
                        }
                    )

            session_delta_json = json.dumps(session_delta, indent=2) if session_delta else "None"

            rendered = render_prompt(
                prompt_name,
                persona=expanded_persona.to_full_description(),
                life_event=life_event0.get("event", ""),
                event_date=life_event0.get("date", ""),
                session_delta=session_delta_json,
                session_id=str(0),
            )

        # ------------------------------------------------------------------
        # evaluation/preference_judge_system
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "evaluation/preference_judge_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # evaluation/preference_judge_user
        # Variables:
        #   required_preferences = JSON of required preferences list
        #   transcript           = JSON of conversation transcript
        #   num_required         = number of required preferences
        # Requires: tasks + an eval run (run test_evaluation.py first).
        # ------------------------------------------------------------------
        elif prompt_name == "evaluation/preference_judge_user":
            eval_result = load_first_eval_result(output_dir)
            if eval_result is None:
                print(f"SKIPPED: {prompt_name} — no eval run found (run test_evaluation.py first)")
                continue
            required_prefs_json = json.dumps(required_preferences, indent=2, ensure_ascii=False)
            transcript = [{"role": t["role"], "content": t["content"]} for t in eval_result["conversation"]]
            transcript_json = json.dumps(transcript, indent=2, ensure_ascii=False)
            rendered = render_prompt(
                prompt_name,
                required_preferences=required_prefs_json,
                transcript=transcript_json,
                num_required=len(required_preferences),
            )

        # ------------------------------------------------------------------
        # evaluation/efficiency_judge_system
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "evaluation/efficiency_judge_system":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # evaluation/efficiency_judge_user
        # Variables:
        #   required_preferences = JSON of required preferences list
        #   transcript           = JSON of conversation transcript
        #   agent_turns          = number of agent turns
        # Requires: tasks + an eval run (run test_evaluation.py first).
        # ------------------------------------------------------------------
        elif prompt_name == "evaluation/efficiency_judge_user":
            eval_result = load_first_eval_result(output_dir)
            if eval_result is None:
                print(f"SKIPPED: {prompt_name} — no eval run found (run test_evaluation.py first)")
                continue
            required_prefs_json = json.dumps(required_preferences, indent=2, ensure_ascii=False)
            transcript = [{"role": t["role"], "content": t["content"]} for t in eval_result["conversation"]]
            transcript_json = json.dumps(transcript, indent=2, ensure_ascii=False)
            agent_turns = sum(1 for t in transcript if t["role"] == "assistant")
            rendered = render_prompt(
                prompt_name,
                required_preferences=required_prefs_json,
                transcript=transcript_json,
                agent_turns=agent_turns,
            )

        # ------------------------------------------------------------------
        # user_simulator/user_simulator_system
        # Variables:
        #   persona_summary      = base_persona (1-line persona string)
        #   required_preferences = bullet list of "id: fact" entries
        # Requires: tasks (run test_task_generation.py first).
        # ------------------------------------------------------------------
        elif prompt_name == "user_simulator/user_simulator_system":
            prefs_bullet = "\n".join(
                f"- {p['id']}: {p['fact']}" for p in required_preferences
            )
            rendered = render_prompt(
                prompt_name,
                persona_summary=task_spec.persona,
                required_preferences=prefs_bullet,
            )

        # ------------------------------------------------------------------
        # user_simulator/user_simulator_user
        # Variable:
        #   conversation = formatted conversation history as a string
        # Uses the session 0 conversation from sessions.json as a realistic
        # stand-in — no eval run needed.
        # ------------------------------------------------------------------
        elif prompt_name == "user_simulator/user_simulator_user":
            conv_lines = []
            for turn in session0.get("conversation", []):
                role = "User" if turn["role"] == "user" else "Agent"
                conv_lines.append(f"{role}: {turn['content']}")
            conversation_str = "\n\n".join(conv_lines)
            rendered = render_prompt(
                prompt_name,
                conversation=conversation_str,
            )
            conversation_str = "\n\n".join(conv_lines)
            rendered = render_prompt(
                prompt_name,
                conversation=conversation_str,
            )

        # ------------------------------------------------------------------
        # agents/agent_system_with_context
        # Variable:
        #   preference_history = formatted preference history from MultiSessionOutput
        # ------------------------------------------------------------------
        elif prompt_name == "agents/agent_system_with_context":
            preference_history = format_preference_history(
                multisession_output,
                include_ids=False,
            )
            rendered = render_prompt(
                prompt_name,
                preference_history=preference_history,
            )

        # ------------------------------------------------------------------
        # agents/agent_system_no_context
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "agents/agent_system_no_context":
            rendered = render_prompt(prompt_name)

        # ------------------------------------------------------------------
        # agents/agent_system_memory
        # No variables — static system prompt.
        # ------------------------------------------------------------------
        elif prompt_name == "agents/agent_system_memory":
            rendered = render_prompt(prompt_name)

        else:
            print(f"WARNING: no variable logic defined for {prompt_name} — skipping")
            continue

        # Save and report
        out_path = save_render(prompt_name, rendered)
        print(f"Rendered: {prompt_name}")
        print(f"  -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
