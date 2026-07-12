# Monthly Characteristic Panel Schema

Sprint 14 uses the uploaded monthly stock-characteristic panel as its primary
empirical dataset. It is treated as a CRSP/Compustat-style panel, not as a
claim that the original paper's complete data construction has been reproduced.

The raw panel may be CSV, TSV, or Parquet. The converter normalizes the fields
below and writes a cleaned Parquet table. Real or licensed data must remain in
`data/private/` or `artifacts/paper_replay/private/`; no proprietary rows are
committed to this repository.

## Required Columns

| Raw column | Cleaned column | Meaning |
| --- | --- | --- |
| `date` | `date` | Month-end observation date |
| `cusip` | `cusip` | Security identifier |
| `permco` | `asset_id` and preserved `permco` | Company/asset identifier |
| `prc` | `price` and preserved `prc` | Price |
| `ret` | `ret` | Monthly return |
| `mktcap` | `market_cap` and preserved `mktcap` | Market capitalization |

The normalized table also retains `cusip`, an internal `asset_id`, and the
source identifier columns where they are available.

## Supported Characteristics

The supported raw characteristic columns are:

```text
beta a2me at ac lme lt_rev ato beme beme_adj c2d c cto e2p idio_vol lev
mktcap lturnover noa oa ol pcm pm dto q high_52w rna roa roe r12_2 r12_7
r2_1 r36_13 s2p sga2s suv
```

`beme_adj` and `c2d` are preserved and reported when present, but are extra
columns and are not included in the default paper-style characteristic set.
The remaining 33 supported fields are used when
`--characteristics all` is selected. With ten bins this produces up to 330
managed portfolios when every characteristic/bin is available.

## Table A1 Mapping

| Cleaned column | Paper Table A1 label |
| --- | --- |
| `a2me` | A2ME |
| `at` | AT |
| `ac` | AC |
| `lme` | LME / Size |
| `lt_rev` | Lt_Rev |
| `ato` | ATO |
| `beme` | BEME |
| `c` | C |
| `cto` | CTO |
| `e2p` | E2P |
| `idio_vol` | Idio vol |
| `lev` | Lev |
| `mktcap` | Mktcap |
| `lturnover` | LTurnover |
| `noa` | NOA |
| `oa` | OA |
| `ol` | OL |
| `pcm` | PCM |
| `pm` | PM |
| `dto` | DTO |
| `q` | Q |
| `high_52w` | Rel to High |
| `rna` | RNA |
| `roa` | ROA |
| `roe` | ROE |
| `r12_2` | r12-2 |
| `r12_7` | r12-7 |
| `r2_1` | r2-1 |
| `r36_13` | r36-13 |
| `s2p` | S2P |
| `sga2s` | SGA2S |
| `suv` | SUV |

`beta` is supported by the uploaded-panel loader but has no separate row in
the mapping list above. `beme_adj` and `c2d` are available extras rather than
required paper-style characteristics.

## Timing Contract

The managed portfolio builder uses the following timing rule:

```text
sort using characteristics at month t
realize the managed portfolio return at month t+1
```

This avoids same-month return lookahead. The exported PCA mapping `V` maps PCA
factor weights to managed-portfolio weights. Recovered weights must therefore
be named `managed_portfolio_weights`, not individual stock weights.

## Validation

```bash
uv run python scripts/validate_monthly_characteristic_panel.py \
  --input data/private/monthly_characteristic_panel.tsv \
  --output-dir artifacts/paper_replay/monthly_panel_validation \
  --format tsv --sep auto
```

The validator reports date coverage, duplicate date/asset rows, return/price/
market-cap coverage, characteristic availability and missing rates, and the
earliest feasible 240-month OOS date. For a panel beginning in 2000, it must
state that exact 2005-start 20-year-lookback replication is impossible and
that a 2005-start shorter-lookback run is pilot-only.
