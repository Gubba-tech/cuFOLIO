# PortOpt QP Extension Release Checklist

## Repository and Provenance

- [ ] Branch is clean: `feature/portopt-unified-qp-cuopt`.
- [ ] Sprint 2-10 tags remain unchanged and are pushed.
- [ ] Sprint 11 provenance commit is recorded and tagged.

## Validation

- [ ] `uv sync --extra dev` completes.
- [ ] Environment smoke test passes.
- [ ] Compileall passes for source, tests, scripts, examples, and benchmarks.
- [ ] Public API, example smoke, and benchmark smoke tests pass.
- [ ] Full CPU suite with `-m "not gpu"` passes.
- [ ] Ruff passes.
- [ ] Existing B40 GPU validation is referenced; any new GPU run is recorded separately.

## Product Artifacts

- [ ] Six examples are present and documented.
- [ ] Six QP notebooks are present.
- [ ] Technical report is present.
- [ ] Advisor demo script is present.
- [ ] PR description/release note is present.
- [ ] Benchmark interpretation note is present.
- [ ] README points to the final documents.
- [ ] Math audit and limitations are updated.
- [ ] `skills/portopt_qp/SKILL.md` and its mirror are synchronized.

## Claims and Scope

- [ ] No unsupported global QP speedup claim is present.
- [ ] No full CRSP/Compustat/IPCA/AP-Trees replication claim is present.
- [ ] Mean-CVaR LP workflow remains separate and untouched.
- [ ] Hard tracking-error constraints are not presented as ordinary QP support.

## Packaging and Security

- [ ] License compatibility is checked.
- [ ] No large generated artifacts are committed accidentally.
- [ ] No credentials, cluster paths, or private data are committed.
- [ ] Benchmark artifacts explain how to regenerate ignored outputs.

