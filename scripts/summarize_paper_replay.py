#!/usr/bin/env python3
"""Summarize paper replay rows and optionally create diagnostic plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _number(value: object) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _annualized_return(values: np.ndarray) -> float | None:
    if values.size == 0 or np.any(values <= -1):
        return None
    return float(np.prod(1.0 + values) ** (12.0 / values.size) - 1.0)


def _metrics_for_group(rows: list[dict[str, str]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "optimal"]
    returns = np.asarray(
        [value for row in successful if (value := _number(row.get("realized_next_return"))) is not None],
        dtype=float,
    )
    annualized_return = _annualized_return(returns)
    annualized_volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(12.0)) if returns.size > 1 else None
    )
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(12.0))
        if returns.size > 1 and np.std(returns, ddof=1) > 0
        else None
    )
    cumulative = np.cumprod(1.0 + returns) if returns.size else np.asarray([])
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0 if returns.size else np.asarray([])
    max_drawdown = float(np.min(drawdown)) if drawdown.size else None
    calmar = (
        annualized_return / abs(max_drawdown)
        if annualized_return is not None and max_drawdown not in (None, 0)
        else None
    )

    def values(name: str) -> list[float]:
        return [value for row in successful if (value := _number(row.get(name))) is not None]

    turnover = values("turnover")
    gross_long = values("gross_long")
    gross_short = values("gross_short")
    violations = values("max_constraint_violation")
    distances = values("weight_l2_distance")
    return {
        "backend": rows[0].get("backend"),
        "model_name": rows[0].get("model_name"),
        "rows": len(rows),
        "successful_rows": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "Sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "Calmar": calmar,
        "average_turnover": float(np.mean(turnover)) if turnover else None,
        "average_gross_long": float(np.mean(gross_long)) if gross_long else None,
        "average_gross_short": float(np.mean(gross_short)) if gross_short else None,
        "max_constraint_violation": max(violations) if violations else None,
        "average_weight_distance_vs_old": (
            float(np.mean(distances)) if distances else None
        ),
        "average_return_difference_vs_old": (
            float(np.mean(values("return_difference_vs_old")))
            if values("return_difference_vs_old")
            else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_cumulative(path: Path, rows: list[dict[str, str]]) -> None:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        value = _number(row.get("realized_next_return"))
        if row.get("status") != "optimal" or value is None:
            continue
        key = f"{row.get('backend')}:{row.get('model_name')}"
        groups[key].append((row.get("rebalance_date", ""), value))
    dates = sorted({date for values in groups.values() for date, _ in values})
    cumulative: dict[str, float] = {key: 1.0 for key in groups}
    by_date = {key: dict(values) for key, values in groups.items()}
    output = []
    for date in dates:
        record: dict[str, object] = {"rebalance_date": date}
        for key in sorted(groups):
            value = by_date[key].get(date)
            if value is not None:
                cumulative[key] *= 1.0 + value
            record[key] = cumulative[key]
        output.append(record)
    _write_csv(path, output)


def _write_summary(path: Path, metrics: list[dict[str, object]]) -> None:
    lines = [
        "# Paper Replay Metrics",
        "",
        "These metrics summarize saved matrix replay windows. They are not full CRSP/Compustat/IPCA empirical replication results.",
        "",
        "| backend | model | success rate | annualized return | annualized volatility | Sharpe | max drawdown | average weight L2 distance | max constraint violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {backend} | {model_name} | {success_rate} | {annualized_return} | "
            "{annualized_volatility} | {Sharpe} | {max_drawdown} | "
            "{average_weight_distance_vs_old} | {max_constraint_violation} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, rows: list[dict[str, str]], metrics: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    cumulative_path = output_dir / "cumulative_returns.csv"
    with cumulative_path.open(encoding="utf-8", newline="") as handle:
        cumulative_rows = list(csv.DictReader(handle))
    columns = [column for column in cumulative_rows[0] if column != "rebalance_date"] if cumulative_rows else []

    if columns:
        figure, axis = plt.subplots()
        dates = [row["rebalance_date"] for row in cumulative_rows]
        for column in columns:
            axis.plot(dates, [float(row[column]) for row in cumulative_rows], label=column)
        axis.set_title("Cumulative replay return")
        axis.legend()
        figure.autofmt_xdate()
        figure.savefig(output_dir / "cumulative_return.png", bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots()
        for column in columns:
            values = np.asarray([float(row[column]) for row in cumulative_rows])
            drawdown = values / np.maximum.accumulate(values) - 1.0
            axis.plot(dates, drawdown, label=column)
        axis.set_title("Replay underwater")
        axis.legend()
        figure.autofmt_xdate()
        figure.savefig(output_dir / "underwater.png", bbox_inches="tight")
        plt.close(figure)

        return_values = []
        return_labels = []
        for column in columns:
            backend, model = column.split(":", maxsplit=1)
            values = []
            for row in rows:
                if (
                    row.get("status") == "optimal"
                    and row.get("backend") == backend
                    and row.get("model_name") == model
                ):
                    values.append(_number(row.get("realized_next_return")))
            return_values.append([value if value is not None else np.nan for value in values])
            return_labels.append(column)
        if return_values:
            width = max(len(values) for values in return_values)
            matrix = np.full((len(return_values), width), np.nan)
            for index, values in enumerate(return_values):
                matrix[index, : len(values)] = values
            figure, axis = plt.subplots()
            image = axis.imshow(matrix, aspect="auto")
            axis.set_yticks(range(len(return_labels)), return_labels)
            axis.set_title("Monthly replay returns")
            figure.colorbar(image, ax=axis)
            figure.savefig(output_dir / "monthly_returns_heatmap.png", bbox_inches="tight")
            plt.close(figure)

    figure, axis = plt.subplots()
    for row in metrics:
        value = row.get("average_weight_distance_vs_old")
        if value is not None:
            axis.bar(f"{row['backend']}:{row['model_name']}", float(value))
    axis.set_title("Average weight distance versus old weights")
    figure.autofmt_xdate()
    figure.savefig(output_dir / "weight_distance.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    for row in metrics:
        value = row.get("max_constraint_violation")
        if value is not None:
            axis.bar(f"{row['backend']}:{row['model_name']}", float(value))
    axis.set_title("Maximum constraint violation")
    figure.autofmt_xdate()
    figure.savefig(output_dir / "constraint_violation.png", bbox_inches="tight")
    plt.close(figure)


def summarize(input_dir: str | Path, plots: bool = False) -> list[dict[str, object]]:
    """Summarize one replay result directory."""
    output_dir = Path(input_dir)
    rows = _read_rows(output_dir / "per_window_results.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("backend", ""), row.get("model_name", ""))].append(row)
    metrics = [_metrics_for_group(group) for _, group in sorted(grouped.items())]
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_cumulative(output_dir / "cumulative_returns.csv", rows)
    _write_summary(output_dir / "summary.md", metrics)
    if plots:
        _write_plots(output_dir, rows, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = summarize(args.input_dir, plots=args.plots)
    print(f"metric_rows={len(metrics)} output_dir={args.input_dir}")


if __name__ == "__main__":
    main()
