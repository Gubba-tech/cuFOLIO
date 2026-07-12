#!/usr/bin/env python3
"""Audit or export saved PortOpt/IPCA factor matrices into replay windows."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from cufolio.qp_paper_replay import PaperReplayWindow, save_replay_window


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            if len(keys) != 1:
                raise ValueError(f"Expected one array in {path}; found {keys}.")
            return np.asarray(data[keys[0]], dtype=float)
    return np.loadtxt(path, delimiter=",")


def _audit_python_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return sorted(names)


def _audit_root(root: Path) -> None:
    if not root.exists():
        print(f"PortOpt root not found: {root}")
        print("No original files can be audited from this path.")
        return
    python_files = sorted(root.rglob("*.py"))
    matrix_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".npy", ".npz", ".parquet", ".csv", ".pkl"}
    )
    print(f"python_files={len(python_files)}")
    for path in python_files:
        names = _audit_python_file(path)
        if names:
            print(f"python: {path.relative_to(root)} functions/classes={names}")
    print(f"candidate_matrix_files={len(matrix_files)}")
    for path in matrix_files:
        print(f"candidate: {path.relative_to(root)}")


def _manual_export(args: argparse.Namespace) -> Path | None:
    manual_paths = (
        args.factor_returns_file,
        args.stock_mapping_file,
        args.factor_covariance_file,
        args.factor_mean_file,
    )
    if not any(manual_paths):
        return None
    if not all(manual_paths):
        raise ValueError(
            "Manual export requires factor returns, stock mapping, factor "
            "covariance, and factor mean files together."
        )
    factor_returns = _load_array(args.factor_returns_file)
    stock_mapping = _load_array(args.stock_mapping_file)
    factor_covariance = _load_array(args.factor_covariance_file)
    factor_mean = _load_array(args.factor_mean_file).reshape(-1)
    if factor_returns.ndim != 2 or stock_mapping.ndim != 2:
        raise ValueError("factor returns and stock mapping must be two-dimensional.")
    date = args.start_date
    window = PaperReplayWindow(
        schema_version="1.0",
        window_id=f"manual_{date}",
        rebalance_date=date,
        model_name="external",
        objective="max_sharpe",
        mapping_mode="factor_space",
        risk_free_rate=0.0,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        short_budget=0.2,
        w_min=-0.08,
        w_max=0.08,
        factor_names=[f"factor_{idx}" for idx in range(factor_mean.size)],
        factor_mean=factor_mean,
        factor_covariance=factor_covariance,
        stock_mapping=stock_mapping,
        notes="Manual external factor matrix export; verify provenance before use.",
    )
    target = args.output_dir / f"{window.window_id}.npz"
    if args.dry_run:
        print(f"would_export={target}")
        print(f"factor_returns_shape={factor_returns.shape}")
        print(f"stock_mapping_shape={stock_mapping.shape}")
        return target
    return save_replay_window(window, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portopt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["PCA", "RP-PCA", "IPCA", "AP-Trees"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--start-date", default="2005-01-31")
    parser.add_argument("--end-date", default="2005-12-31")
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--factor-returns-file", type=Path, default=None)
    parser.add_argument("--stock-mapping-file", type=Path, default=None)
    parser.add_argument("--factor-covariance-file", type=Path, default=None)
    parser.add_argument("--factor-mean-file", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"models={args.models} k_values={args.k_values}")
    print(f"date_range={args.start_date}..{args.end_date}")
    _audit_root(args.portopt_root)
    manual_target = _manual_export(args)
    if manual_target is not None:
        return
    if args.dry_run:
        print("dry_run=true; no replay windows were written")
        print("manual input flags are available for supplied factor matrices")
        return
    raise SystemExit(
        "No export adapter is available for unknown original artifacts. "
        "Use the manual factor input flags after auditing the source matrices."
    )


if __name__ == "__main__":
    main()

