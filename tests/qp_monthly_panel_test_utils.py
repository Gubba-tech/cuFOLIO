from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/monthly_panel/monthly_characteristic_panel_sample.tsv"


def expanded_monthly_panel(path: Path, months: int = 16, assets: int = 12) -> Path:
    """Expand the compact uploaded-style fixture into a deterministic panel."""
    base = pd.read_csv(FIXTURE, sep="\t")
    rows = []
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    for month_index, date in enumerate(dates):
        for asset_index in range(assets):
            row = base.iloc[asset_index % len(base)].copy()
            row["date"] = date
            row["cusip"] = f"{asset_index + 1:08d}"
            row["permco"] = asset_index + 1
            row["prc"] = 10.0 + asset_index
            row["ret"] = 0.002 * (asset_index - 4) + 0.001 * (month_index % 3)
            row["mktcap"] = 100.0 + 10.0 * asset_index
            for column in base.columns[5:]:
                if column != "mktcap":
                    row[column] = float(asset_index + 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    output = path / "monthly_characteristic_panel.tsv"
    frame.to_csv(output, sep="\t", index=False)
    return output
