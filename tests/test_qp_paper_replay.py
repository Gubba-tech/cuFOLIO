import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qp_test_utils import require_cuopt

from cufolio.qp_paper_replay import (
    compute_replay_diagnostics,
    load_replay_window,
    solve_replay_window,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/paper_replay/synthetic_pca_k3_window.npz"


def test_load_replay_window_validates_synthetic_fixture():
    window = load_replay_window(FIXTURE)

    assert window.schema_version == "1.0"
    assert window.model_name == "PCA"
    assert window.mapping_mode == "factor_space"
    assert window.n_assets == 16
    assert window.n_factors == 3
    assert window.factor_mean.shape == (3,)
    assert window.stock_mapping.shape == (16, 3)
    assert window.realized_next_returns.shape == (16,)


def test_solve_replay_window_osqp_and_diagnostics():
    window = load_replay_window(FIXTURE)
    result = solve_replay_window(window, backend="osqp")
    diagnostics = compute_replay_diagnostics(window.old_weights, result, window)

    assert result.solution.status == "optimal"
    assert result.stock_weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert diagnostics["objective_value"] is not None
    assert diagnostics["weight_l2_distance"] is not None
    assert diagnostics["realized_next_return"] is not None
    assert diagnostics["max_constraint_violation"] <= 1e-5
    assert diagnostics["box_violation"] <= 1e-6


def test_replay_and_summary_scripts_work_on_fixture(tmp_path):
    result_dir = tmp_path / "results"
    replay = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_paper_replay.py"),
            "--input-dir",
            str(FIXTURE.parent),
            "--output-dir",
            str(result_dir),
            "--backend",
            "osqp",
            "--compare-old",
            "--write-summary",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0, replay.stderr
    assert "replay_rows=1" in replay.stdout
    assert (result_dir / "per_window_results.csv").exists()
    assert (result_dir / "per_window_results.jsonl").exists()
    assert (result_dir / "summary.md").exists()

    summary = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_paper_replay.py"),
            "--input-dir",
            str(result_dir),
            "--plots",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert summary.returncode == 0, summary.stderr
    assert (result_dir / "metrics.csv").exists()
    assert (result_dir / "cumulative_returns.csv").exists()
    assert (result_dir / "cumulative_return.png").exists()
    assert (result_dir / "underwater.png").exists()
    assert (result_dir / "monthly_returns_heatmap.png").exists()
    assert (result_dir / "weight_distance.png").exists()
    assert (result_dir / "constraint_violation.png").exists()


@pytest.mark.gpu
def test_solve_replay_window_cuopt_skips_without_gpu():
    require_cuopt()
    window = load_replay_window(FIXTURE)
    result = solve_replay_window(window, backend="cuopt")
    assert result.solution.status == "optimal"
    assert np.isfinite(result.solution.objective_value)
