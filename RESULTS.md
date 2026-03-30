# Results

## v1

Evaluation across 50 sessions, 5 tasks per session, 3 eval runs per task. Scores are macro-averaged: run mean → task mean → session mean → overall mean.

- **Model**: gpt-4.1
- **Memories retrieved per turn**: 5

| Agent | Preference Recall | Stale Recall Rate | Task Completion (score=1.0) | Avg Memory Tokens / Eval |
|-------|-------------------|-------------------|-----------------------------|--------------------------|
| context | 0.85 | 3.6% | 47.9% | - |
| foundry | 0.64 | 8.1% | 11.3% | 3472 |
| aws | 0.62 | 11.9% | 8.8% | 3586 |
| google | 0.55 | 13.5% | 6.3% | 1741 |
| mem0 | 0.33 | 16.4% | 0.3% | 376 |
| nocontext | 0.07 | 2.0% | 0.0% | - |

## v1 (memory budget = 100 tokens)

Evaluation across 50 sessions, 5 tasks per session, 3 eval runs per task. Memory token budget = 100 tokens (with minimum 1 memory which can go over 100 tokens). Scores are macro-averaged: run mean → task mean → session mean → overall mean.

- **Model**: gpt-4.1
- **Memory token budget**: 100

| Agent | Preference Recall | Stale Recall Rate | Task Completion (score=1.0) | Avg Memory Tokens / Eval |
|-------|-------------------|-------------------|-----------------------------|--------------------------|
| foundry | 0.39 | 9.4% | 0.8% | 708 |
| aws | 0.38 | 12.5% | 1.1% | 600 |
| google | 0.37 | 10.4% | 1.3% | 475 |
| mem0 | 0.33 | 14.8% | 0.5% | 345 |
