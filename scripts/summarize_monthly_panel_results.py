#!/usr/bin/env python3
"""Create Table-2-style metrics, diagnostics, and plots for replay results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _number(value: object) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _read_rows(input_dir: str | Path) -> list[dict[str, str]]:
    input_dir = Path(input_dir)
    candidates = [input_dir / "per_window_results.csv", input_dir / "replay" / "per_window_results.csv"]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"No per_window_results.csv found below {input_dir}")


def _annualized_return(values: np.ndarray) -> float | None:
    if values.size == 0 or np.any(values <= -1):
        return None
    return float(np.prod(1.0 + values) ** (12.0 / values.size) - 1.0)


def _yearly_returns(dates: list[pd.Timestamp], values: np.ndarray) -> dict[int, float]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for date, value in zip(dates, values):
        by_year[date.year].append(float(value))
    return {
        year: float(np.prod(1.0 + np.asarray(months)) - 1.0)
        for year, months in by_year.items()
        if np.all(np.asarray(months) > -1)
    }


def metrics_for_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """Compute one Table-2-style metrics row for one backend/model group."""
    successful = [row for row in rows if row.get("status") == "optimal"]
    dated_values = [
        (pd.Timestamp(row.get("rebalance_date")), _number(row.get("realized_next_return")))
        for row in successful
        if row.get("rebalance_date") and _number(row.get("realized_next_return")) is not None
    ]
    dated_values.sort(key=lambda item: item[0])
    dates = [item[0] for item in dated_values]
    returns = np.asarray([item[1] for item in dated_values], dtype=float)
    cumulative = np.cumprod(1.0 + returns) if returns.size else np.asarray([])
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0 if cumulative.size else np.asarray([])
    yearly = _yearly_returns(dates, returns)

    def values(name: str) -> list[float]:
        return [
            value
            for row in successful
            if (value := _number(row.get(name))) is not None
        ]

    status_counts = pd.Series([row.get("status", "") for row in rows]).value_counts().to_dict()
    nonoptimal_non_skipped = sum(
        count
        for status, count in status_counts.items()
        if status not in {"optimal", "skipped"}
    )
    cagr = _annualized_return(returns)
    max_drawdown = float(np.min(drawdown)) if drawdown.size else None
    monthly_sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1))
        if returns.size > 1 and np.std(returns, ddof=1) > 0
        else None
    )
    annualized_sharpe = monthly_sharpe * math.sqrt(12.0) if monthly_sharpe is not None else None
    return {
        "backend": rows[0].get("backend", "") if rows else "",
        "model_name": rows[0].get("model_name", "") if rows else "",
        "start_period": str(dates[0].date()) if dates else None,
        "end_period": str(dates[-1].date()) if dates else None,
        "time_in_market": int(returns.size),
        "time_in_market_fraction": float(returns.size / len(rows)) if rows else 0.0,
        "total_windows": len(rows),
        "optimal_count": int(status_counts.get("optimal", 0)),
        "skipped_count": int(status_counts.get("skipped", 0)),
        "failed_count": int(nonoptimal_non_skipped),
        "other_status_count": int(nonoptimal_non_skipped - status_counts.get("failed", 0)),
        "cagr": cagr,
        "annualized_return": cagr,
        "annualized_volatility": (
            float(np.std(returns, ddof=1) * math.sqrt(12.0)) if returns.size > 1 else None
        ),
        "monthly_sharpe": monthly_sharpe,
        "annualized_sharpe": annualized_sharpe,
        # Keep the historical field used by grid artifacts and downstream code.
        "Sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "Calmar": cagr / abs(max_drawdown) if cagr is not None and max_drawdown not in (None, 0) else None,
        "best_month": float(np.max(returns)) if returns.size else None,
        "worst_month": float(np.min(returns)) if returns.size else None,
        "best_year": max(yearly.values()) if yearly else None,
        "worst_year": min(yearly.values()) if yearly else None,
        "average_gross_long": float(np.mean(values("gross_long"))) if values("gross_long") else None,
        "average_gross_short": float(np.mean(values("gross_short"))) if values("gross_short") else None,
        "average_turnover": float(np.mean(values("turnover"))) if values("turnover") else None,
        "maximum_constraint_violation": max(values("max_constraint_violation"), default=None),
        "maximum_box_violation": max(values("box_violation"), default=None),
        "maximum_short_budget_violation": max(values("short_budget_violation"), default=None),
        "status_counts": "; ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _returns_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    records = []
    for row in rows:
        value = _number(row.get("realized_next_return"))
        if row.get("status") == "optimal" and value is not None:
            records.append(
                {
                    "date": pd.Timestamp(row["rebalance_date"]),
                    "series": f"{row.get('backend')}:{row.get('model_name')}",
                    "return": value,
                }
            )
    if not records:
        return pd.DataFrame(columns=["date"])
    return (
        pd.DataFrame(records)
        .pivot(index="date", columns="series", values="return")
        .sort_index()
        .reset_index()
    )


def _write_diagnostics(output_dir: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "backend",
        "model_name",
        "rebalance_date",
        "window_id",
        "status",
        "skip_reason",
        "error",
        "max_constraint_violation",
        "box_violation",
        "short_budget_violation",
        "sum_weights",
        "gross_long",
        "gross_short",
        "solve_time",
        "total_time",
    ]
    _write_csv(
        output_dir / "constraint_diagnostics.csv",
        [{field: row.get(field) for field in fields} for row in rows],
    )
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("backend", ""), row.get("model_name", ""))].append(row)
    weights = []
    for group_rows in groups.values():
        metric = metrics_for_rows(group_rows)
        weights.append(
            {
                key: metric.get(key)
                for key in (
                    "backend",
                    "model_name",
                    "average_gross_long",
                    "average_gross_short",
                    "average_turnover",
                    "maximum_constraint_violation",
                    "maximum_box_violation",
                    "maximum_short_budget_violation",
                )
            }
        )
    _write_csv(output_dir / "weights_summary.csv", weights)


def _write_markdown(path: Path, metrics: list[dict[str, object]]) -> None:
    columns = [
        "backend",
        "model_name",
        "start_period",
        "end_period",
        "time_in_market",
        "cagr",
        "annualized_volatility",
        "monthly_sharpe",
        "annualized_sharpe",
        "Sharpe",
        "max_drawdown",
        "Calmar",
        "best_month",
        "worst_month",
        "best_year",
        "worst_year",
        "average_gross_long",
        "average_gross_short",
        "average_turnover",
        "maximum_constraint_violation",
        "optimal_count",
        "skipped_count",
        "failed_count",
    ]
    lines = [
        "# Monthly Panel Empirical Metrics",
        "",
        "These are empirical pilot metrics from saved replay rows. They are not full paper replication results.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in metrics:
        lines.append("| " + " | ".join(str(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, str]], returns: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated: list[str] = []
    if returns.empty:
        return generated
    dates = returns["date"].to_numpy()
    series = [column for column in returns.columns if column != "date"]
    cumulative = returns.copy()
    for column in series:
        cumulative[column] = (1.0 + returns[column].fillna(0.0)).cumprod()
    figure, axis = plt.subplots(figsize=(10, 5))
    for column in series:
        axis.plot(dates, cumulative[column], label=column)
    axis.set_title("Cumulative managed-portfolio QP returns")
    axis.legend()
    figure.autofmt_xdate()
    figure.savefig(output_dir / "cumulative_returns.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("cumulative_returns.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    for column in series:
        values = cumulative[column].to_numpy(dtype=float)
        axis.plot(dates, values / np.maximum.accumulate(values) - 1.0, label=column)
    axis.set_title("Underwater curve")
    axis.legend()
    figure.autofmt_xdate()
    figure.savefig(output_dir / "underwater.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("underwater.png")

    matrix = returns[series].to_numpy(dtype=float).T
    figure, axis = plt.subplots(figsize=(10, max(3, len(series))))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set_yticks(range(len(series)), series)
    axis.set_title("Monthly realized returns")
    figure.colorbar(image, ax=axis)
    figure.savefig(output_dir / "monthly_returns_heatmap.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("monthly_returns_heatmap.png")

    diagnostics = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10, 5))
    for key, group in diagnostics.groupby(["backend", "model_name"], dropna=False):
        values = pd.to_numeric(group["max_constraint_violation"], errors="coerce")
        axis.plot(group["rebalance_date"], values, marker=".", label=f"{key[0]}:{key[1]}")
    axis.set_title("Maximum constraint violation")
    axis.legend()
    figure.autofmt_xdate()
    figure.savefig(output_dir / "constraint_violation.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("constraint_violation.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    for key, group in diagnostics.groupby(["backend", "model_name"], dropna=False):
        axis.plot(group["rebalance_date"], pd.to_numeric(group["gross_long"], errors="coerce"), label=f"{key[0]}:{key[1]} long")
        axis.plot(group["rebalance_date"], -pd.to_numeric(group["gross_short"], errors="coerce"), linestyle="--", label=f"{key[0]}:{key[1]} short")
    axis.set_title("Gross exposure")
    axis.legend()
    figure.autofmt_xdate()
    figure.savefig(output_dir / "gross_exposure.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("gross_exposure.png")
    return generated


def summarize_run(input_dir: str | Path, plots: bool = False) -> list[dict[str, object]]:
    """Write empirical outputs into ``input_dir`` and return metric rows."""
    output_dir = Path(input_dir)
    rows = _read_rows(output_dir)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("backend", ""), row.get("model_name", ""))].append(row)
    metrics = [metrics_for_rows(group) for _, group in sorted(grouped.items())]
    _write_csv(output_dir / "metrics_table.csv", metrics)
    _write_markdown(output_dir / "metrics_table.md", metrics)
    returns = _returns_frame(rows)
    returns.to_csv(output_dir / "monthly_returns.csv", index=False, date_format="%Y-%m-%d")
    if returns.empty:
        pd.DataFrame(columns=["date"]).to_csv(output_dir / "cumulative_returns.csv", index=False)
    else:
        cumulative = returns.copy()
        for column in [column for column in returns.columns if column != "date"]:
            cumulative[column] = (1.0 + returns[column].fillna(0.0)).cumprod()
        cumulative.to_csv(output_dir / "cumulative_returns.csv", index=False, date_format="%Y-%m-%d")
    _write_diagnostics(output_dir, rows)
    generated = _write_plots(output_dir, rows, returns) if plots else []
    (output_dir / "plot_manifest.txt").write_text(
        "\n".join(generated) + ("\n" if generated else ""), encoding="utf-8"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = summarize_run(args.input_dir, plots=args.plots)
    print(f"metric_rows={len(metrics)} output_dir={args.input_dir}")


if __name__ == "__main__":
    main()
