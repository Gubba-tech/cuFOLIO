# Unified QP Extension Limitations

The current extension is a validated cuFOLIO QP integration and product demo
layer. The following boundaries are intentional:

- It does not claim full CRSP/Compustat data preparation or replication.
- It does not reimplement or claim full RP-PCA, IPCA, or AP-Trees research
  pipelines. Those workflows can supply external `factor_returns` and stock
  mapping `V` through the adapter API.
- Hard tracking-error bounds are not represented as ordinary QPs. Tracking
  error is currently supported as a quadratic objective penalty.
- Per-asset turnover limits are not implemented; only a total turnover budget
  is supported.
- The examples and current validation artifacts make no QP speedup claim.
  Speedup statements require dedicated benchmark scripts and saved artifacts.
- Sprint 9 benchmark inputs are deterministic synthetic/public data. Their
  observed timing ratios are scoped to the generated artifact and hardware;
  they are not general performance claims.
- Benchmark artifacts do not establish real-data or rolling-production
  performance, and they do not add CRSP/Compustat/IPCA/AP-Trees replication.
- `backend="cuopt"` requires a working cuOpt GPU runtime. It never silently
  falls back to OSQP or another CPU solver.
- Factor-space mode is explicit. It must be selected with
  `mapping_mode="factor_space"`; stock-space behavior remains the default.
