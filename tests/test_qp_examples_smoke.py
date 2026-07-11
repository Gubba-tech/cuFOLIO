import json
import subprocess
import sys
from pathlib import Path

import pytest
from qp_test_utils import require_cuopt

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run_example(name, *args):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{name} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for marker in ("status:", "objective:", "max_constraint_violation:"):
        assert marker in result.stdout
    return result.stdout


@pytest.mark.parametrize(
    "name, args",
    [
        ("qp_min_variance_quickstart.py", ("--n-assets", "6")),
        ("qp_mean_variance_quickstart.py", ("--n-assets", "6")),
        (
            "qp_max_sharpe_regularized_long_short.py",
            ("--n-assets", "10"),
        ),
        (
            "qp_factor_space_pca_demo.py",
            ("--n-assets", "10", "--n-factors", "2"),
        ),
        (
            "qp_external_factor_adapter_demo.py",
            ("--n-assets", "8", "--n-factors", "2"),
        ),
        ("qp_vs_cvar_baseline_overview.py", ()),
    ],
)
def test_qp_examples_run_on_cpu(name, args):
    _run_example(name, "--backend", "osqp", *args)


@pytest.mark.gpu
def test_qp_min_variance_example_runs_on_cuopt():
    require_cuopt()
    _run_example(
        "qp_min_variance_quickstart.py",
        "--backend",
        "cuopt",
        "--n-assets",
        "6",
    )


@pytest.mark.gpu
def test_qp_max_sharpe_example_runs_on_cuopt():
    require_cuopt()
    _run_example(
        "qp_max_sharpe_regularized_long_short.py",
        "--backend",
        "cuopt",
        "--n-assets",
        "10",
    )


def test_qp_demo_notebooks_are_valid_lightweight_json():
    notebook_dir = ROOT / "notebooks" / "portopt_qp"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    assert len(notebooks) == 6
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
