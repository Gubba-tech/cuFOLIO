from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from qp_monthly_panel_test_utils import expanded_monthly_panel

ROOT = Path(__file__).resolve().parents[1]


def _run_pilot(input_path: Path, output_dir: Path, backend: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_monthly_panel_pca_pilot.py"),
            "--monthly-panel",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--format",
            "tsv",
            "--sep",
            "tab",
            "--start-date",
            "2000-12-31",
            "--end-date",
            "2001-04-30",
            "--lookback-months",
            "12",
            "--k-values",
            "3",
            "--characteristics",
            "beta,a2me,beme,lme,r12_2",
            "--n-bins",
            "3",
            "--weighting",
            "value",
            "--min-assets-per-bin",
            "2",
            "--lambda-l1",
            "1.7e-4",
            "--lambda-l2",
            "1e-3",
            "--backend",
            backend,
            "--max-windows",
            "1",
            "--allow-short-lookback",
            "--assume-characteristics-lagged",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_monthly_panel_pca_pilot_exports_and_replays(tmp_path):
    input_path = expanded_monthly_panel(tmp_path, months=16, assets=12)
    output_dir = tmp_path / "pilot"
    _run_pilot(input_path, output_dir, "both")

    assert (output_dir / "processed/monthly_characteristic_panel.parquet").exists()
    assert (output_dir / "managed_portfolios/managed_portfolio_returns.parquet").exists()
    assert list((output_dir / "windows_pca").glob("*.npz"))
    assert (output_dir / "pilot_manifest.json").exists()
    with (output_dir / "replay/per_window_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["backend"] for row in rows} == {"osqp", "cuopt"}
    assert {row["status"] for row in rows}.issubset({"optimal", "skipped", "failed"})
    assert any(row["backend"] == "osqp" and row["status"] == "optimal" for row in rows)


def test_monthly_panel_pilot_cuopt_request_skips_without_fallback(tmp_path):
    input_path = expanded_monthly_panel(tmp_path, months=16, assets=12)
    output_dir = tmp_path / "cuopt_pilot"
    _run_pilot(input_path, output_dir, "cuopt")

    with (output_dir / "replay/per_window_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["backend"] == "cuopt"
    assert rows[0]["status"] in {"optimal", "skipped"}
    if rows[0]["status"] == "skipped":
        assert "cuOpt" in rows[0]["skip_reason"] or "cuopt" in rows[0]["skip_reason"]
