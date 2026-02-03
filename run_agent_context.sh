#!/bin/bash
set -e

echo "=== Stage 1: Data Generation ==="
SESSION_DIR=$(uv run python test_data_generation.py | grep '^SESSION_DIR=' | cut -d= -f2)
echo "Using session: $SESSION_DIR"
echo ""

echo "=== Stage 2: Task Generation ==="
uv run python test_task_generation.py --session "$SESSION_DIR" --count 3
echo ""

echo "=== Stage 3a: Evaluation (context agent) ==="
uv run python test_evaluation.py --session "$SESSION_DIR" --task all --agent context
echo ""

echo "=== Stage 3b: Evaluation (nocontext agent) ==="
uv run python test_evaluation.py --session "$SESSION_DIR" --task all --agent nocontext
echo ""

echo "=== Pipeline complete ==="
