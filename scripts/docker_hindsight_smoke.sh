#!/usr/bin/env bash
# Run the Hindsight smoke test inside an Ubuntu 24.04 Docker container.
# Uses --rm to auto-remove the container on exit. No orphaned images
# beyond the base ubuntu:24.04 which is cached.
#
# Tests only the Hindsight store (populate + retrieve), NOT the full evaluation
# pipeline which requires DefaultAzureCredential (unavailable in Docker).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Acquiring Azure token on host ==="
AZURE_TOKEN=$(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv)
if [ -z "$AZURE_TOKEN" ]; then
    echo "ERROR: Failed to get Azure token. Run 'az login' first."
    exit 1
fi
echo "Token acquired (${#AZURE_TOKEN} chars)"

echo "=== Running Hindsight smoke test in Docker (Ubuntu 24.04) ==="

docker run \
    --rm \
    --name hindsight-smoke-test \
    -v "${REPO_DIR}:/app" \
    --env-file "${REPO_DIR}/.env" \
    -e "AZURE_TOKEN=${AZURE_TOKEN}" \
    -w /app \
    ubuntu:24.04 \
    bash -c '
set -euo pipefail

echo "--- System info ---"
ldd --version 2>&1 | head -1

echo "--- Installing system dependencies ---"
apt-get update -qq && apt-get install -y -qq curl python3 python3-venv libxml2 sudo > /dev/null 2>&1

echo "--- Creating non-root user (PostgreSQL initdb refuses to run as root) ---"
useradd -m -s /bin/bash testuser

# Run the rest as non-root user
su - testuser -c "
set -euo pipefail

# Re-export env vars (su drops them)
export AZURE_TOKEN=\"${AZURE_TOKEN}\"
export HOME=/home/testuser

# Source .env file
set -a
source /app/.env
set +a

echo \"--- Installing uv ---\"
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH=\"/home/testuser/.local/bin:\$PATH\"

echo \"--- Creating isolated venv (not touching host .venv) ---\"
export UV_PROJECT_ENVIRONMENT=/tmp/hindsight-venv
cd /app
uv sync --extra hindsight 2>&1 | tail -5

echo \"--- Running Hindsight store smoke test ---\"
uv run python scripts/smoke_test_hindsight.py
"
'
