#!/usr/bin/env python3
"""Run saved PortOpt-style QP windows through the cuFOLIO compiler."""

from __future__ import annotations

import argparse
from pathlib import Path

from cufolio.qp_paper_replay import run_replay_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--compare-old", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_replay_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        models=args.models,
        max_windows=args.max_windows,
        compare_old=args.compare_old,
        write_summary=args.write_summary,
        workers=args.workers,
    )
    print(f"replay_rows={len(rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
