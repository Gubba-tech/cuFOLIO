# QP Benchmark Artifacts

Sprint 9 benchmark outputs are generated locally or on Slurm under a
timestamped directory in `artifacts/benchmarks/`.

Large raw CSV, JSONL, and environment artifacts are ignored by default. Small
reviewed summaries may be committed under `artifacts/benchmarks/reviewed/`.
Generated artifacts include environment metadata, the git commit, backend,
compile/build/solve/postprocess timings, feasibility diagnostics, and any
OSQP/cuOpt comparison fields available for the same problem.

The generated summaries use the wording `observed speed ratio in this
benchmark run`. They are not universal speedup claims.
