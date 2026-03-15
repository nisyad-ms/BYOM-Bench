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
