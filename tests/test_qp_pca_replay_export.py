import subprocess
import sys
from pathlib import Path

import pandas as pd

from cufolio.qp_paper_data import build_amp_universe, build_managed_portfolios
from cufolio.qp_paper_replay import load_replay_window

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/fixtures/paper_data"


def _run(command, timeout=120):
    result = subprocess.run(
        [sys.executable, *map(str, command)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


def test_pca_exporter_and_replay_work_on_synthetic_pipeline(tmp_path):
    universe, _ = build_amp_universe(
        monthly_returns=DATA / "monthly_returns.parquet",
        characteristics=DATA / "characteristics_monthly.parquet",
        start_date="2004-01-31",
        end_date="2005-12-31",
        lookback_months=12,
    )
    universe_path = tmp_path / "universe.parquet"
    universe.to_parquet(universe_path, index=False)
    managed, _ = build_managed_portfolios(
        monthly_returns=DATA / "monthly_returns.parquet",
        characteristics=DATA / "characteristics_monthly.parquet",
        universe=universe_path,
        n_bins=5,
        lag_months=1,
        start_date="2004-01-31",
        end_date="2005-12-31",
    )
    managed_path = tmp_path / "managed.parquet"
    managed.to_parquet(managed_path, index=False)
    windows = tmp_path / "windows"
    _run(
        [
            ROOT / "scripts/export_pca_replay_windows.py",
            "--managed-portfolio-returns",
            managed_path,
            "--output-dir",
            windows,
            "--k-values",
            "3",
            "--start-date",
            "2005-01-31",
            "--end-date",
            "2005-12-31",
            "--lookback-months",
            "12",
            "--lambda-l1",
            "0.01",
            "--lambda-l2",
            "0.01",
            "--max-windows",
            "3",
        ]
    )
    window_paths = sorted(windows.glob("*.npz"))
    assert len(window_paths) == 3
    window = load_replay_window(window_paths[0])
    assert window.model_name == "PCA"
    assert window.mapping_mode == "factor_space"
    assert window.n_factors == 3
    assert "managed-portfolio" in window.notes

    results = tmp_path / "results"
    _run(
        [
            ROOT / "scripts/run_paper_replay.py",
            "--input-dir",
            windows,
            "--output-dir",
            results,
            "--backend",
            "osqp",
            "--write-summary",
        ]
    )
    _run([ROOT / "scripts/summarize_paper_replay.py", "--input-dir", results])
    rows = pd.read_csv(results / "per_window_results.csv")
    assert rows.shape[0] == 3
    assert set(rows["status"]) == {"optimal"}
    assert (results / "metrics.csv").exists()
    assert (results / "cumulative_returns.csv").exists()

