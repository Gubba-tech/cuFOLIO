# PortOpt/IPCA Replay Audit

Date: 2026-07-11

## Source and Availability

The cuFOLIO workspace does not contain `../PortOpt_IPCA`. The public GitHub
repository was cloned temporarily for this audit at commit
`8b2a8a890364a435139bc74cefc8911f5cbfb129`:

```text
https://github.com/Gubba-tech/PortOpt_IPCA
```

The clone contains source code and documentation, but no saved `.npz`, `.npy`,
Parquet, or CSV replay matrices, old portfolio-weight files, or Table 2 result
artifact. The repository's runtime data files referenced by the code, such as
`asness_final_revisedmkt.csv`, `asness_withprc.csv`, and
`df_all_withprc.csv`, are not present in the clone.

Therefore Sprint 12 can implement a replay interface and synthetic parity
fixture, but it cannot run a real PCA/IPCA 2005 pilot from the repository alone.

## Matrix and Factor Locations

### QP solver and objective inputs

`PortOpt_factor/optimizer/pyport.py` contains the original QP path:

- `sigMatShrinkage` computes the l2 covariance modification and accepts a factor mapping.
- `constrain_matrix` builds fully invested, long-short, max-Sharpe, box, l1, turnover, benchmark, and factor-exposure rows.
- `penalty_vector` builds the linear l1 and max-Sharpe objective vector.
- `sigMat_expend` expands the covariance block and appends the max-Sharpe scale variable.
- `portfolio_optimization` assembles the OSQP problem, solves it, and recovers max-Sharpe weights by dividing the decision block by the final scale.

The old function receives `meanVec` and `sigMat` from the caller. It does not
persist those matrices. In factor mode, `meanVec` and `sigMat` are factor mean
and factor covariance, while `factor` is the stock-to-factor mapping used by
the constraints and l1 split.

### PCA and RP-PCA

`PortOpt_factor/optimizer/pca_rppca.py` contains `PCA_factor`,
`PRPCA_factor`, and `RPPCAOOS`. In each rolling window, the code computes
factor scores and a stock mapping (`Lambdahat`), then records:

```text
factor covariance = np.cov(Fhat.T)
factor mean       = np.mean(Fhat, axis=0)
```

The QP call uses `factor=Lambdahat[:, :K]`. Out-of-sample returns are formed by
mapping the factor portfolio back through `Lambdahat` and its Gram inverse.

The default example call uses `gamma=15`, `K=6`, and `window=240`.

### IPCA

`PortOpt_factor/optimizer/ipca.py` contains `IPCA_factor` and `IPCAOOS`.
`IPCA_factor` selects the rolling date window, fits `InstrumentedPCA`, and
returns `Gamma`, factor scores, next-period returns, next-period characteristics,
and the last-window characteristics. `IPCAOOS` then computes:

```text
Lambdahat_IPCA = inv(Gamma.T @ X_last.T @ X_last @ Gamma) @ Gamma.T @ X_last.T
factor covariance = np.cov(factors_temp, rowvar=True)
factor mean       = np.mean(factors_temp, axis=1)
```

The QP receives the factor mapping derived from `Lambdahat_IPCA.T` and the
factor mean/covariance. The output individual-stock weights are recovered with
the same mapping. The default call uses `gamma=15`, `K=6`, and `window=240`.

### AP-Trees

The clone's optimizer code references AP-Trees as a portfolio-model input, but
does not contain a complete AP-Trees estimator or saved AP-Trees matrices. A
future AP-Trees replay must therefore provide externally saved factor returns,
mapping, mean, and covariance.

## Tuning Grid and Dates

Both `RPPCAOOS` and `IPCAOOS` define:

```python
g1 = np.exp(np.linspace(np.log(1e-6), np.log(5), 10))
g2 = np.exp(np.linspace(np.log(1e-6), np.log(5), 10))
```

The source calls max-Sharpe with `longShort=0.2`, `maxAlloc=0.08`, and
`riskfree=0`. The paper-style default rolling window is 240 monthly
observations, with the out-of-sample exercise beginning in 2005 and ending in
2022. The exact dates are supplied by the input data's unique date sequence;
the source code does not hard-code a complete date manifest.

## Stored Outputs

The functions return `SR_grid`, return paths, and in the IPCA case `w_df` and
`w_adj` in memory. The clone does not serialize those outputs. This is the
reason for the Sprint 12 replay artifact format: export must happen while the
original data pipeline is available, without copying proprietary raw data into
cuFOLIO.

## Audit Conclusion

The old code exposes enough mathematical inputs for matrix replay, but the
current public clone lacks the raw data and saved matrices needed for an actual
PCA/IPCA pilot. Sprint 12 therefore treats a supplied replay artifact as the
contract boundary. It does not reimplement IPCA and does not claim empirical
parity until old matrices, dates, and realized returns are supplied.

