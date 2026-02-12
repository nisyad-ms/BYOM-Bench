# Plan: Simplify Task + User Prompt + User Simulator Logic

## Context

The current evaluation pipeline generates an **evaluation task** (scenario + 6 preferences via LLM), then a **user_prompt** (opening message via a second LLM call), then hands both to the **user simulator** which drives the conversation. In practice, the scenario is too narrow for all 6 preferences, so the simulator topic-bridges away from it anyway. The user_prompt is a redundant LLM call whose only job is to not leak preferences — something the simulator prompt already handles.

This change removes the evaluation scenario/task concept entirely and lets the user simulator drive the full conversation from the opening message onward. Task generation reduces from 2 LLM calls to 1 (preference selection only).

## Changes

### 1. Schema changes — `memory_gym/schemas.py`

**Replace `EvaluationTask` with `EvaluationTaskSpec`:**
```python
@dataclass
class EvaluationTaskSpec:
    task_id: str
    rubric: EvaluationRubric
    persona: str              # base persona string
    preference_history: str   # output of format_preference_history()
```
- `to_dict()` / `from_dict()` — straightforward serialization; `from_dict()` should silently ignore old-format keys (`evaluation_event`, `event_type`, `reasoning`) for backward compat
- Delete `EvaluationTask` class (lines 672-722)
- Delete the `user_prompt` property, `scenario_type`, `reasoning` fields

**Update `MultiSessionEvaluationResult`:**
- Remove `evaluation_event: LifeEvent | None` field
- Keep `rubric: EvaluationRubric | None`
- Remove `evaluation_event` from `to_dict()` / `from_dict()`

### 2. Task generator — `memory_gym/task_generators/evaluation_task.py`

**Update constants:**
```python
MIN_EVOLVED_PREFS = 2          # was 3
EVOLVED_PREF_RATIO = 1/3       # was 0.5
```

**Simplify `generate_batch()`:**
- Replace `previous_events: list[str]` with `used_preference_ids: list[str]` tracking all previously-selected preference IDs across tasks
- After each task, append selected pref IDs to `used_preference_ids`
- Remove `generated_events` tracking

**Replace `_generate_single_task()` internals:**
- Call new `_select_preferences()` instead of `_generate_evaluation_event_v2()` + `_generate_user_prompt()`
- Call `summarize_events()` + `format_preference_history()` once, pass `preference_history` into `EvaluationTaskSpec`
- Return `EvaluationTaskSpec`

**New `_select_preferences()` method:**
- Single LLM call using new prompt (see step 5)
- Takes `preference_history`, `num_evolved`, `num_baseline`, `used_preference_ids`
- Returns `list[dict]` of selected preference dicts (no scenario, no event)

**Delete methods:**
- `_generate_evaluation_event_v2()` (lines 267-340)
- `_generate_evaluation_event()` (lines 387-457)
- `_generate_user_prompt()` (lines 342-385)
- `_create_persona_summary()` (already deleted this session)

**Update convenience functions** (`generate_evaluation_task`, `generate_evaluation_tasks`, `generate_evaluation_tasks_parallel`):
- Change param `previous_events` → `used_preference_ids`
- Return `EvaluationTaskSpec` instead of `EvaluationTask`

### 3. User simulator — `memory_gym/evaluation_multisession/user_simulator.py`

**Constructor:** Accept `EvaluationTaskSpec` instead of `EvaluationTask`

**`_build_system_prompt()`:**
- Pass `preference_history=self.task.preference_history` instead of `evaluation_event`
- Pass `persona_summary=self.task.persona`

**`get_initial_message()`:** Generate opening message via LLM call instead of returning pre-generated `task.user_prompt`:
```python
def get_initial_message(self) -> tuple[str, str | None]:
    messages = [
        {"role": "system", "content": self._system_prompt},
        {"role": "user", "content": "Generate your opening message to the AI assistant."},
    ]
    response = self.client.complete_chat(messages=messages, ...)
    clean, scratchpad = self._extract_scratchpad(response.strip())
    return clean, scratchpad
```
Note: return type changes to `tuple[str, str | None]` to capture the initial scratchpad.

### 4. Runner — `memory_gym/evaluation_multisession/runner.py`

- Update `eval_task` type from `EvaluationTask` → `EvaluationTaskSpec`
- Update `get_initial_message()` call to handle new return type `(message, scratchpad)`
- Remove log lines referencing `eval_task.evaluation_event.event`
- `summarize_events()` call at line 147 stays — it's used by `ContextAwareAgent`, not the simulator

### 5. Judge — `memory_gym/evaluation_multisession/judge.py`

- Update `evaluation_task` type from `EvaluationTask` → `EvaluationTaskSpec`
- Remove `evaluation_event` from `_combine_results()` and result constructor
- Remove `LifeEvent` import

### 6. Prompts

**New: `task_generation/preference_selection_instruction.yaml`**
- LLM selects 2 evolved + 4 baseline preferences with interesting interactions
- Input: `preference_evolution_story`, `num_evolved_required`, `num_baseline_required`, `used_preference_ids`
- Output: `{"selected_preferences": [...], "reasoning": "..."}`
- No scenario generation, just preference selection with diversity constraints

**New: `evaluation/user_simulator_system_v6.yaml`**
- Remove `{evaluation_event}` variable, add `{preference_history}` variable
- Remove STARTING STATE section ("joining conversation in progress")
- Add OPENING MESSAGE section: simulator generates its own natural opening that doesn't reveal preferences
- Scratchpad tracking starts from opening message
- Keep TOPIC BRIDGING and preference testing rules
- Incorporate key anti-leakage rules from `user_prompt_generator_system_v2.yaml`

**Delete:**
- `task_generation/user_prompt_generator_system_v2.yaml`
- `task_generation/user_prompt_generator_system.yaml` (v1 if exists)
- `task_generation/evaluation_task_instruction_v6.yaml` (replaced by preference_selection_instruction)

### 7. Test scripts and config

**`test_task_generation.py`:**
- Replace `get_existing_events()` with `get_used_preference_ids()` that reads `rubric.required_preferences[*].id` from existing task files
- Update `generate_tasks()` to pass `used_preference_ids` instead of `previous_events`

**`test_evaluation.py`:**
- `EvaluationTask.from_dict()` → `EvaluationTaskSpec.from_dict()`

**`configs/prompt_config.yaml`:**
- Add `preference_selection_instruction` (no version suffix)
- Update `user_simulator_system` version to `v6`
- Remove `evaluation_task_instruction`, `user_prompt_generator_system`

**`gather_results.py`:** No changes needed (only reads `rubric.required_preferences`)

### 8. `memory_gym/task_generators/__init__.py`

- Update exports — same function names, types change under the hood

## Files to modify
| File | Action |
|------|--------|
| `memory_gym/schemas.py` | Replace `EvaluationTask` with `EvaluationTaskSpec`, update `MultiSessionEvaluationResult` |
| `memory_gym/task_generators/evaluation_task.py` | Major refactor: 1 LLM call, new `_select_preferences()`, delete 3 methods |
| `memory_gym/task_generators/__init__.py` | Update exports |
| `memory_gym/evaluation_multisession/user_simulator.py` | New `get_initial_message()`, use `preference_history` |
| `memory_gym/evaluation_multisession/runner.py` | Type updates, handle new `get_initial_message()` return |
| `memory_gym/evaluation_multisession/judge.py` | Type updates, remove `evaluation_event` |
| `memory_gym/prompts/task_generation/preference_selection_instruction.yaml` | **NEW** |
| `memory_gym/prompts/evaluation/user_simulator_system_v6.yaml` | **NEW** |
| `memory_gym/prompts/task_generation/user_prompt_generator_system_v2.yaml` | **DELETE** |
| `memory_gym/prompts/task_generation/evaluation_task_instruction_v6.yaml` | **DELETE** |
| `configs/prompt_config.yaml` | Update prompt references |
| `test_task_generation.py` | `get_existing_events()` → `get_used_preference_ids()` |
| `test_evaluation.py` | Import + deserialization update |

## Verification
1. Run `ruff check .` and `ruff format .` to verify no lint errors
2. Run `uv run python test_task_generation.py --count 3` — should produce task files with new schema (no `evaluation_event`, has `preference_history`)
3. Inspect generated task JSON to verify preference selection works and diversity is maintained
4. Ask before running evaluation (expensive)
