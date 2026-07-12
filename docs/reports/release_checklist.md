# PortOpt QP Extension Release Checklist

## Repository and Provenance

- [x] Branch is clean: `feature/portopt-unified-qp-cuopt`.
- [x] Sprint 2-10 tags remain unchanged and are pushed.
- [x] Sprint 11 provenance commit is recorded and tagged.

## Validation

- [x] `uv sync --extra dev` completes.
- [x] Environment smoke test passes.
- [x] Compileall passes for source, tests, scripts, examples, and benchmarks.
- [x] Public API, example smoke, and benchmark smoke tests pass.
- [x] Full CPU suite with `-m "not gpu"` passes.
- [x] Ruff passes.
- [x] Existing B40 GPU validation is referenced; any new GPU run is recorded separately.

## Product Artifacts

- [x] Six examples are present and documented.
- [x] Six QP notebooks are present.
- [x] Technical report is present.
- [x] Advisor demo script is present.
- [x] PR description/release note is present.
- [x] Benchmark interpretation note is present.
- [x] README points to the final documents.
- [x] Math audit and limitations are updated.
- [x] `skills/portopt_qp/SKILL.md` and its mirror are synchronized.

## Claims and Scope

- [x] No unsupported global QP speedup claim is present.
- [x] No full CRSP/Compustat/IPCA/AP-Trees replication claim is present.
- [x] Mean-CVaR LP workflow remains separate and untouched.
- [x] Hard tracking-error constraints are not presented as ordinary QP support.

## Packaging and Security

- [x] License compatibility is checked: documentation-only changes use the repository Apache-2.0 license.
- [x] No large generated artifacts are committed accidentally.
- [x] No credentials, cluster paths, or private data are committed.
- [x] Benchmark artifacts explain how to regenerate ignored outputs.
