# Sprint 5 Max-Sharpe QP Validation

Sprint 5 implementation and validation were completed on commit `8c0e651`
(`Implement Sprint 5 max-Sharpe QP`) on branch
`feature/portopt-unified-qp-cuopt`.

CPU validation:

```text
147 passed, 2 skipped, 35 deselected
```

GPU validation:

- Slurm job: `45981`
- State: `COMPLETED`, exit code `0:0`
- Partition: `debug-b40x4`
- Node: `b40x4-05`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Driver: `595.71.05`
- CUDA: `13.2`
- cuFOLIO extra: `cuda13`
- GPU pytest: `10 passed, 24 deselected`

The validated scope covers stock-level max-Sharpe reparameterization, scaled
box and long-short constraints, l1/l2/l1+l2 regularization, generic `V`
compiler/small-QP behavior, positive-excess feasibility validation, and
optimizer result recovery fields.

This validation makes no QP speedup claim and does not claim end-to-end
IPCA/PCA/AP-Trees workflow support.
