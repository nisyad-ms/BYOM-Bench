# Results

## 2026-03-02 (v0.1, v0.2)

Evaluation across 250 unique samples (50 personas, 5 tasks per persona, 3 eval runs per task). Scores are macro-averaged across the 3 eval runs and then averaged across the 5 tasks, then 50 personas.

- **Model**: gpt-4.1
- **Memories retrieved per turn**: 5

| Agent | Preference Recall | Task Completion Rate |
|-------|-------------------|---------------------|
| perfect_memory | 0.90 | 69.2% |
| foundry_local | 0.71 | 25.7% |
| aws | 0.64 | 15.7% |
| google | 0.55 | 9.2% |
| no_memory | 0.02 | 0.0% |


## 2026-03-05 (v0.3)

Evaluation across 250 unique samples (50 personas, 5 tasks per persona, 3 eval runs per task = 750 evals per agent). Scores are hierarchical macro-averages: runs → tasks → sessions → overall.

- **Model**: gpt-4.1
- **Memories retrieved per turn**: 5

| Agent | Preference Recall | Task Completion Rate |
|-------|-------------------|---------------------|
| perfect_memory | 0.90 | 67.6% |
| foundry_local | 0.64 | 19.5% |
| aws | 0.55 | 13.5% |
| google | 0.49 | 9.7% |
| no_memory | 0.09 | 0.1% |