# Development Environment

Use an isolated environment for QP development. Do not rely on cluster/system site packages when they contain a broken `pandas` / `pyarrow` installation.

## No uv Found on HPC

First check that the active Python is new enough for cuFOLIO:

```bash
python --version
```

cuFOLIO requires Python `>=3.11`. If `uv` is missing on an HPC login node, install it user-local:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Alternative user-local install:

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If system Python is polluted by broken `pyarrow` / `pandas` packages, isolate the shell before creating the environment:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

Fallback without `uv`:

```bash
python3.11 -m venv .venv --clear
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
python scripts/smoke_qp_env.py
python -m compileall -q src tests scripts
pytest tests/test_qp_compiled_convention.py -q
pytest tests/test_qp_osqp_validation_backend.py -q
pytest -m "not gpu" -q
```

## CPU Validation Environment

From the cuFOLIO repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/smoke_qp_env.py
python -m compileall -q src tests
pytest tests/test_qp_compiled_convention.py
pytest tests/test_qp_osqp_validation_backend.py
```

If your shell automatically exposes user or system site packages, isolate Python further:

```bash
export PYTHONNOUSERSITE=1
```

## GPU / cuOpt Environment

Install the CUDA extra matching the machine:

```bash
python -m pip install -e ".[dev,cuda12]"
```

or:

```bash
python -m pip install -e ".[dev,cuda13]"
```

Then run:

```bash
python scripts/smoke_qp_env.py
pytest -m gpu tests/test_qp_cuopt_backend.py
```

## Conda Alternative

```bash
conda create -n cufolio-qp python=3.11
conda activate cufolio-qp
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/smoke_qp_env.py
```

## Notes

- A system-level `pyarrow` failure is an environment problem, not a cuFOLIO QP code failure, if the isolated environment passes `scripts/smoke_qp_env.py`.
- CPU validation uses OSQP through CVXPY only when `QPParameters(backend="osqp")` is selected.
- Production GPU QP solves must use cuOpt directly and must not silently fall back to CPU.
