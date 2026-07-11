#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts examples
uv run pytest tests/test_qp_examples_smoke.py -q
uv run pytest -m "not gpu" -q
