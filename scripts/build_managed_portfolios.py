#!/usr/bin/env python3
"""Build lagged characteristic-sorted managed portfolio returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cufolio.qp_paper_data import build_managed_portfolios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-returns", type=Path, required=True)
    parser.add_argument("--characteristics", type=Path, required=True)
    parser.add_argument("--universe-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--characteristics-list", nargs="+", default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--weighting", choices=("equal", "value"), default="equal")
    parser.add_argument("--lag-months", type=int, default=6)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_characteristics = args.characteristics_list or "all"
    returns, metadata = build_managed_portfolios(
        monthly_returns=args.monthly_returns,
        characteristics=args.characteristics,
        universe=args.universe_file,
        selected_characteristics=selected_characteristics,
        n_bins=args.n_bins,
        weighting=args.weighting,
        lag_months=args.lag_months,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(args.output_dir / "managed_portfolio_returns.parquet", index=False)
    (args.output_dir / "mapping_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Managed Portfolio Summary",
        "",
        f"Rows: {returns.shape[0]}",
        f"Portfolios: {returns['portfolio_id'].nunique() if not returns.empty else 0}",
        "",
        "These portfolios are managed-portfolio inputs for PCA/RP-PCA replay; they are not individual stock weights.",
        "",
    ]
    (args.output_dir / "managed_portfolio_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"rows={returns.shape[0]} portfolios={returns['portfolio_id'].nunique() if not returns.empty else 0}")


if __name__ == "__main__":
    main()

