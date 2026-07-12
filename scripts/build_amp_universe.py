#!/usr/bin/env python3
"""Build a configurable AMP-style selected universe by rebalance date."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_paper_data import build_amp_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-returns", type=Path, required=True)
    parser.add_argument("--characteristics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--market-cap-coverage", type=float, default=0.90)
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.20)
    parser.add_argument("--lookback-months", type=int, default=240)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected, summary = build_amp_universe(
        monthly_returns=args.monthly_returns,
        characteristics=args.characteristics,
        start_date=args.start_date,
        end_date=args.end_date,
        market_cap_coverage=args.market_cap_coverage,
        min_price=args.min_price,
        max_missing_fraction=args.max_missing_fraction,
        lookback_months=args.lookback_months,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output_dir / "universe_by_date.parquet", index=False)
    summary.to_csv(args.output_dir / "universe_summary.csv", index=False)
    lines = [
        "# AMP-Style Universe Summary",
        "",
        "Filters are configurable approximations; they are not a claim of exact paper universe replication.",
        "",
        summary.to_string(index=False) if not summary.empty else "No selected rows.",
        "",
    ]
    (args.output_dir / "universe_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"dates={summary.shape[0]} selected_rows={selected.shape[0]}")


if __name__ == "__main__":
    main()
