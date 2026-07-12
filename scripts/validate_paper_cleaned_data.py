#!/usr/bin/env python3
"""Validate cleaned paper-style portfolio data and write a report."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_paper_data import validate_cleaned_data, write_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-returns", type=Path, default=None)
    parser.add_argument("--daily-returns", type=Path, default=None)
    parser.add_argument("--characteristics", type=Path, default=None)
    parser.add_argument("--benchmark-returns", type=Path, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/paper_replay/data_validation_report.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_cleaned_data(
        monthly_returns=args.monthly_returns,
        daily_returns=args.daily_returns,
        characteristics=args.characteristics,
        benchmark_returns=args.benchmark_returns,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_months=args.lookback_months,
    )
    write_validation_report(report, args.report_path)
    print(f"status={report['status']} report={args.report_path}")
    print(f"issues={len(report['issues'])}")


if __name__ == "__main__":
    main()

