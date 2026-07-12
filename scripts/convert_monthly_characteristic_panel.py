#!/usr/bin/env python3
"""Convert a raw monthly characteristic panel to cleaned Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_monthly_panel import convert_monthly_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper_replay/processed/monthly_characteristic_panel.parquet"),
    )
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-id-col", default="permco")
    parser.add_argument("--cusip-col", default="cusip")
    parser.add_argument("--return-col", default="ret")
    parser.add_argument("--price-col", default="prc")
    parser.add_argument("--market-cap-col", default="mktcap")
    parser.add_argument("--characteristics", default="all")
    parser.add_argument("--drop-missing-ret", action="store_true")
    parser.add_argument("--min-price", type=float)
    parser.add_argument(
        "--winsorize-characteristics",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
    )
    parser.add_argument("--standardize-characteristics", choices=("cross_sectional",))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, metadata = convert_monthly_panel(
        input_path=args.input,
        output_path=args.output,
        format=args.format,
        sep=args.sep,
        date_col=args.date_col,
        asset_id_col=args.asset_id_col,
        cusip_col=args.cusip_col,
        return_col=args.return_col,
        price_col=args.price_col,
        market_cap_col=args.market_cap_col,
        characteristics=args.characteristics,
        drop_missing_ret=args.drop_missing_ret,
        min_price=args.min_price,
        winsorize_characteristics=(
            tuple(args.winsorize_characteristics)
            if args.winsorize_characteristics
            else None
        ),
        standardize_characteristics=args.standardize_characteristics,
    )
    print(f"rows={metadata['rows']} output={args.output}")


if __name__ == "__main__":
    main()
