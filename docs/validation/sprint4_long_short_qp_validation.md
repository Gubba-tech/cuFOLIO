# Sprint 4 Long-Short QP Validation

Sprint 4 validates long-short budget constraints on the
`feature/portopt-unified-qp-cuopt` branch.

CPU validation:

```text
118 passed, 2 skipped, 25 deselected
```

GPU validation:

- Slurm job: `45435`
- State: `COMPLETED`, exit code `0:0`
- Partition: `debug-b40x4`
- Node: `b40x4-02`
- CUDA: `13.2`
- cuFOLIO extra: `cuda13`
- GPU pytest: `11 passed, 18 deselected`

The validated scope covers explicit `pos`/`neg` long-short splits for stock-level
minimum-variance, mean-variance, target-return, l1, l2-squared, and l1 plus
l2-squared QPs, plus generic `V` compiler/small-QP cases. `short_budget=0` was
checked against direct long-only bounds.

This validation makes no QP speedup claim.
