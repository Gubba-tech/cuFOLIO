# Sprint 6 Friction-Constraint Validation

Sprint 6 implementation was completed on commit `26afda9`
(`Validate Sprint 6 friction constraints`) on branch
`feature/portopt-unified-qp-cuopt`.

CPU validation:

```text
186 passed, 2 skipped, 55 deselected
```

GPU validation:

- Slurm job: `45995`
- State: `COMPLETED`, exit code `0:0`
- Partition: `debug-b40x4`
- Node: `b40x4-05`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Driver: `595.71.05`
- CUDA: `13.2`
- cuFOLIO extra: `cuda13`
- GPU pytest: `20 passed, 39 deselected`

The validated scope covers total turnover budgets, benchmark l1 exposure
budgets, linear factor exposure bounds, tracking-error quadratic penalties,
ordinary and max-Sharpe scaled formulations, generic `V` compiler/small-QP
behavior, and medium-size combined cases.

Per-asset turnover limits and hard tracking-error constraints remain future
work. This validation makes no QP speedup claim and does not claim end-to-end
IPCA/PCA/AP-Trees workflow support.
