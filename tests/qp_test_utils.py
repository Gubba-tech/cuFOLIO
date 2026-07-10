import importlib.util
import shutil
import subprocess

import numpy as np
import pytest

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp

GPU_SKIP_REASON = "cuOpt GPU runtime unavailable; QP GPU test skipped."


def small_returns_dict():
    return {
        "mean": np.array([0.030, 0.050, 0.020, 0.040]),
        "covariance": np.array(
            [
                [0.050, 0.006, 0.004, 0.002],
                [0.006, 0.080, 0.005, 0.003],
                [0.004, 0.005, 0.040, 0.004],
                [0.002, 0.003, 0.004, 0.030],
            ]
        ),
    }


def medium_returns_dict(n_assets: int = 20):
    rng = np.random.default_rng(123)
    design = rng.normal(size=(n_assets, n_assets))
    covariance = design.T @ design + 1e-3 * np.eye(n_assets)
    mean = rng.normal(size=n_assets) * 0.01
    return {"mean": mean, "covariance": covariance}


def require_cuopt():
    if not cuopt_gpu_runtime_available():
        pytest.skip(GPU_SKIP_REASON)


def cuopt_gpu_runtime_available() -> bool:
    if importlib.util.find_spec("cuopt") is None:
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def assert_feasible_solution(compiled, solution, tol: float = 1e-6):
    assert solution.status == "optimal"
    assert solution.max_constraint_violation <= tol
    assert compiled.objective_value(solution.x) == pytest.approx(
        solution.objective_value,
        abs=max(tol, 1e-6),
        rel=1e-6,
    )


def relative_objective_gap(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(1.0, abs(reference))


def assert_cuopt_matches_osqp(
    compiled,
    *,
    objective_tol: float = 1e-4,
    constraint_tol: float = 1e-5,
):
    require_cuopt()
    osqp_solution = solve_compiled_qp_osqp(compiled)
    cuopt_solution = solve_compiled_qp_cuopt(compiled)

    assert_feasible_solution(compiled, osqp_solution, tol=constraint_tol)
    assert_feasible_solution(compiled, cuopt_solution, tol=constraint_tol)
    assert (
        relative_objective_gap(
            compiled.objective_value(cuopt_solution.x),
            compiled.objective_value(osqp_solution.x),
        )
        <= objective_tol
    )
    np.testing.assert_allclose(
        compiled.recover_stock_weights(cuopt_solution.x),
        compiled.recover_stock_weights(osqp_solution.x),
        atol=1e-4,
        rtol=1e-4,
    )
    return osqp_solution, cuopt_solution
