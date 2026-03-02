# Results

## 2026-03-02

Evaluation across 250 unique samples (50 personas, 5 tasks per persona, 3 eval runs per task). Scores are macro-averaged across the 3 eval runs and then averaged across the 5 tasks, then 50 personas.

| Agent | Preference Recall | Turn Efficiency |
|-------|------------|-----------|
| perfect_memory | 0.90 | 0.96 |
| foundry_local | 0.72 | 0.86 |
| google | 0.55 | 0.78 |
| no_memory | 0.02 | 0.66 |
| aws | 0.63* | 0.82* |

\* ~80 samples are still running for AWS.
