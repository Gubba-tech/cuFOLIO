# Sprint 7 Factor-Space Workflow Validation

Sprint 7 implementation was completed on commit `62e9074`
(`Implement Sprint 7 factor-space QP workflows`) on branch
`feature/portopt-unified-qp-cuopt`.

CPU validation:

```text
205 passed, 2 skipped, 59 deselected
```

GPU validation:

- Slurm job: `46031`
- State: `COMPLETED`, exit code `0:0`
- Partition: `debug-b40x4`
- Node: `b40x4-05`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Driver: `595.71.05`
- CUDA: `13.2`
- cuFOLIO extra: `cuda13`
- GPU pytest: `4 passed, 14 deselected`

The validated scope covers explicit true factor-space mode, factor-space
ordinary and max-Sharpe QPs, stock constraints through `V @ z`, factor-space
tracking-error penalties with stock covariance, deterministic PCA data, and
external factor-return/mapping adapters.

RP-PCA, IPCA, and AP-Trees are supported only through externally supplied
factor returns and mappings. Full CRSP/Compustat or end-to-end IPCA/AP-Trees
replication is not claimed. This validation makes no QP speedup claim.
