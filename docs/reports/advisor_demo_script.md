# Five-Minute Advisor Demo Script

## 1. One-Sentence Pitch

This project adds a PortOpt-style unified QP track to NVIDIA cuFOLIO, with a
direct cuOpt backend for stock and factor-space portfolio optimization while
leaving the existing Mean-CVaR LP workflow intact.

## 2. What Changed from the NVIDIA cuFOLIO Baseline

The baseline is centered on scenario generation and Mean-CVaR portfolio
optimization. The extension adds explicit QP objects, a compiler, an OSQP
validation path, and a direct cuOpt QP path. The two workflows remain separate;
the QP extension does not replace Mean-CVaR.

## 3. What Changed from the Original PortOpt Paper

The paper supplies the mathematical source for unified objectives,
regularization, long-short constraints, mappings, and max-Sharpe
reparameterization. The cuFOLIO project packages those ideas as a public
library extension, adds direct cuOpt integration, and adds deterministic PCA
and external-factor adapters. It does not claim full CRSP/Compustat, IPCA, or
AP-Trees empirical replication.

## 4. Three CPU Commands

From the repository root, run:

```bash
uv run python examples/qp_min_variance_quickstart.py --backend osqp
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend osqp
uv run python examples/qp_factor_space_pca_demo.py --backend osqp
```

These show a basic stock-level QP, a constrained max-Sharpe QP, and a true
factor-space PCA QP. The default is explicit OSQP so the demo is runnable on a
CPU login host.

## 5. One Max-Sharpe GPU Example

On a B40 or H200 node, select the CUDA extra that matches `nvidia-smi` and
request cuOpt explicitly:

```bash
uv sync --extra cuda13 --extra dev
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend cuopt
```

The backend selection is intentional. If cuOpt is unavailable, the command
raises a GPU-backend error; it does not substitute OSQP.

## 6. One Factor-Space PCA Example

`qp_factor_space_pca_demo.py` builds deterministic PCA factor data, solves in
factor coordinates, and recovers stock weights through `V`. This demonstrates
that factor space is a real compiler mode, not only a label on a stock-space
problem.

## 7. One Benchmark Artifact Explanation

The benchmark runners write environment metadata, dimensions, statuses,
feasibility diagnostics, compile/build/solve/postprocess timings, and total
time. `summary.md` reports an observed OSQP/cuOpt ratio only when matching
problems succeeded on both backends. The Sprint 10 B40 artifact is a scoped
correctness and timing record, not a universal speedup claim.

## 8. What Is Not Claimed

We do not claim global QP speedups, full CRSP/Compustat replication, full IPCA
or AP-Trees estimation, hard tracking-error constraints as ordinary QPs, or
changes to the existing Mean-CVaR LP workflow.

## 9. Next Research Steps

The natural follow-ups are a separately scoped public-data rolling-window demo,
optional full IPCA/AP-Trees replication, review of matching GPU artifacts, and
an upstream cuFOLIO PR or short workshop/thesis appendix.

