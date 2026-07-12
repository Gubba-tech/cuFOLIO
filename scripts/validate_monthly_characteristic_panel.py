#!/usr/bin/env python3
"""Validate the uploaded CRSP/Compustat-style monthly stock panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_monthly_panel import (
    validate_monthly_panel,
    write_monthly_panel_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_replay/monthly_panel_validation"),
    )
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-id-col", default="permco")
    parser.add_argument("--cusip-col", default="cusip")
    parser.add_argument("--return-col", default="ret")
    parser.add_argument("--price-col", default="prc")
    parser.add_argument("--market-cap-col", default="mktcap")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-months", type=int, default=240)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_monthly_panel(
        path=args.input,
        format=args.format,
        sep=args.sep,
        date_col=args.date_col,
        asset_id_col=args.asset_id_col,
        cusip_col=args.cusip_col,
        return_col=args.return_col,
        price_col=args.price_col,
        market_cap_col=args.market_cap_col,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_months=args.lookback_months,
    )
    summary = write_monthly_panel_validation(report, args.output_dir)
    print(f"status={report['status']} summary={summary}")
    if report["status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
