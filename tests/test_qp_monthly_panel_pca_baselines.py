from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_monthly_panel_pca_baselines import _configs, _safe_solve_job  # noqa: E402


def test_baseline_runner_records_infeasible_window_instead_of_aborting():
    config = next(config for config in _configs() if config["kind"] == "pca")
    history = np.full((12, 8), -0.01)
    next_returns = np.full(8, -0.01)

    row = _safe_solve_job((config, "2000-01-31", history, next_returns, "osqp"))

    assert row["status"] == "failed"
    assert row["model_name"] == "pca_k6_unregularized"
    assert "QPCompilationError" in row["error"]
