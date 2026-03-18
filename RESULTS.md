# Results

## 2026-03-02 (v0.1, v0.2)

Evaluation across 250 unique samples (50 personas, 5 tasks per persona, 3 eval runs per task). Scores are macro-averaged across the 3 eval runs and then averaged across the 5 tasks, then 50 personas.

- **Model**: gpt-4.1
- **Memories retrieved per turn**: 5

| Agent | Preference Recall | Task Completion Rate |
|-------|-------------------|---------------------|
| perfect_memory | 0.90 | 69.2% |
| foundry_local | 0.71 | 25.7% |
| aws | - | - |
| google | 0.55 | 9.2% |
| no_memory | 0.02 | 0.0% |

## 2026-03-17 (v0.4)

Evaluation across 50 sessions, 5 tasks per session, 3 eval runs per task. Scores are macro-averaged: run mean → task mean → session mean → overall mean.

- **Model**: gpt-4.1
- **Memories retrieved per turn**: 5
- **Eval run**: `2026-03-16_215407`

| Agent | Preference Recall | Stale Recall Rate | Task Completion (score=1.0) | Avg Memory Tokens / Eval |
|-------|-------------------|-------------------|-----------------------------|--------------------------|
| context | 0.85 | 3.6% | 47.9% | - |
| foundry_local | 0.64 | 8.1% | 11.3% | 3472 |
| aws | 0.62 | 11.9% | 8.8% | 3586 |
| google | 0.55 | 13.5% | 6.3% | 1741 |
| mem0 | 0.33 | 16.4% | 0.3% | 376 |
| nocontext | 0.07 | 2.0% | 0.0% | - |

## 2026-03-17 (v0.4, memory budget=100 tokens)

Evaluation across 50 sessions, 5 tasks per session, 3 eval runs per task. Memory token budget = 100 token (with minimum 1 memory which can go over 100 tokens). Scores are macro-averaged: run mean → task mean → session mean → overall mean.

- **Model**: gpt-4.1
- **Memory token budget**: 100
- **Eval run**: `2026-03-17_093532`

| Agent | Preference Recall | Stale Recall Rate | Task Completion (score=1.0) | Avg Memory Tokens / Eval |
|-------|-------------------|-------------------|-----------------------------|--------------------------|
| foundry_local | 0.39 | 9.4% | 0.8% | 708 |
| aws | 0.38 | 12.5% | 1.1% | 600 |
| google | 0.37 | 10.4% | 1.3% | 475 |
| mem0 | 0.33 | 14.8% | 0.5% | 345 |
