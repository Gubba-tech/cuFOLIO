# Sprint 3 l1 QP Validation

Sprint 3 was validated on commit `27127c6` (`Validate Sprint 3 l1 QP regularization`).

CPU validation:

```text
100 passed, 2 skipped, 14 deselected
```

GPU validation:

- Slurm job: `45434`
- Partition: `debug-b40x4`
- Node: `b40x4-02`
- CUDA: `13.2`
- cuFOLIO extra: `cuda13`
- GPU pytest: `6 passed, 10 deselected`

This validation establishes correctness for the Sprint 3 l1 compiler and direct cuOpt
path. It makes no QP speedup claim.
