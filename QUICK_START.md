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

## 3. Evaluate your memory on the pre-generated dataset

Download pre-generated data from [Azure Storage](https://memorypublicdata001wus2.blob.core.windows.net/internal-benchmark/outputs) and place it in `outputs/`. Then run evaluation directly:

```bash
# Run evaluation on all datasets in outputs/
uv run python scripts/test_evaluation.py --outputs-dir outputs --session all --agent foundry

# Gather results
uv run python scripts/gather_results.py --outputs-dir outputs
```

## 4. Full pipeline: generate conversations and evaluate

```bash
# Stage 1: Generate conversation data (1 random persona, 2 sessions — takes a few minutes)
# For a rich evolution of user preferences, use ~10 sessions
uv run python scripts/test_data_generation.py --outputs-dir outputs --sessions 2

# Stage 2: Generate 3 evaluation tasks
uv run python scripts/test_task_generation.py --outputs-dir outputs --count 3

# Stage 3: Run evaluation with foundry agent
uv run python scripts/test_evaluation.py --outputs-dir outputs --session all --agent foundry

# Stage 4: Gather results into Excel
uv run python scripts/gather_results.py --outputs-dir outputs
```

Results are saved to `outputs/`. See [README.md](README.md) for the full reference.
