#!/usr/bin/env python3
"""Build paper-style characteristic-sorted monthly managed portfolios."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_monthly_panel import build_paper_managed_portfolios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-panel", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_replay/managed_portfolios_monthly_panel"),
    )
    parser.add_argument("--characteristics", default="all")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--weighting", choices=("equal", "value"), default="value")
    parser.add_argument("--market-cap-col", default="market_cap")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--min-assets-per-bin", type=int, default=5)
    parser.add_argument("--universe-method", choices=("all", "amp"), default="all")
    parser.add_argument("--market-cap-coverage", type=float, default=0.90)
    parser.add_argument("--min-price", type=float)
    parser.add_argument("--max-missing-fraction", type=float, default=1.0)
    parser.add_argument("--assume-characteristics-lagged", action="store_true")
    parser.add_argument(
        "--sort-at-t-return-at-t-plus-1",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    returns, _, metadata = build_paper_managed_portfolios(
        monthly_panel=args.monthly_panel,
        output_dir=args.output_dir,
        characteristics=args.characteristics,
        n_bins=args.n_bins,
        weighting=args.weighting,
        market_cap_col=args.market_cap_col,
        start_date=args.start_date,
        end_date=args.end_date,
        min_assets_per_bin=args.min_assets_per_bin,
        universe_method=args.universe_method,
        market_cap_coverage=args.market_cap_coverage,
        min_price=args.min_price,
        max_missing_fraction=args.max_missing_fraction,
        assume_characteristics_lagged=args.assume_characteristics_lagged,
        sort_at_t_return_at_t_plus_1=args.sort_at_t_return_at_t_plus_1,
    )
    print(
        f"rows={len(returns)} portfolios={metadata['managed_portfolios_created']} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
