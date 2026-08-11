# Execution diagnostics

Only job 63094 and execution SHA
`f00c45ae3d571a1eabe9e66d6034b90683f83626` contribute timing rows to the
final benchmark.

- Job 63018 used the first execution contract and was discarded after a
  native cuOpt process abort. No timing row from this attempt was reused.
- Job 63026 completed the second execution contract, but 52 diagonal-QP
  cuOpt rows returned `NumericalError`. The entire attempt was discarded.
- Job 63080 localized the diagonal-QP failure to the cuDSS ADAT search
  direction. Job 63083 verified that cuOpt's augmented KKT form resolves the
  failure. Job 63084 showed that applying augmented KKT globally is unsuitable
  for the LP-MIXED family.
- The frozen v3 contract therefore uses `augmented=1` only for
  QP-DIAGONAL. Final job 63094 produced all 480 expected cold rows with strict
  solver-optimal status and no timeout or process failure.

All 360 registered solver rows pass their per-solver canonical reconstruction
and original-constraint gates. The paired cross-solver objective gate accepts
168 of 180 registered pairs, corresponding to 56 of 60 canonical instances.
The 12 excluded pairs are the three repetitions of four LP-RANGED instances;
they are omitted from every speedup median and are not converted into capped
timings.
