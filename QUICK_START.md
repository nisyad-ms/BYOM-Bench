# Quick Start

## 1. Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set your Azure OpenAI credentials (required):

```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
AZURE_OPENAI_DEPLOYMENTS="gpt-4.1-001,gpt-4.1-002"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
```

Then authenticate:

```bash
az login
```

## 3. Run the pipeline

```bash
# Stage 1: Generate conversation data (1 persona, 2 sessions — takes a few minutes)
# For a rich evolution of user preferences, use ~10 sessions
uv run python scripts/test_data_generation.py --persona single --sessions 2

# Stage 2: Generate 3 evaluation tasks
uv run python scripts/test_task_generation.py --count 3

# Stage 3: Run evaluation with foundry_local agent
uv run python scripts/test_evaluation.py --session all --agent foundry_local

# Stage 4: Gather results into Excel
uv run python scripts/gather_results.py
```

Results are saved to `outputs/`. See [README.md](README.md) for the full reference.

## 4. Evaluate an existing dataset

Download pre-generated data from [Azure Storage](https://memorypublicdata001wus2.blob.core.windows.net/internal-benchmark/outputs) and place it in `outputs/`. Then run evaluation directly:

```bash
# Run evaluation on all datasets in outputs/
uv run python scripts/test_evaluation.py --session all --agent foundry_local

# Gather results
uv run python scripts/gather_results.py
```

## 5. Bring your own memory

Want to evaluate a custom memory backend? See [BRING_YOUR_OWN_MEMORY.md](BRING_YOUR_OWN_MEMORY.md) for how to implement and register a new memory store.
