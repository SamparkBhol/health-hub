#!/usr/bin/env python3
"""Fit and rolling-origin evaluate the public HMIS environmental model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.forecasting.public_hmis import write_model


def main() -> int:
    report = write_model()
    pooled = report["pooled"]
    print(
        "Public HMIS malaria model: "
        f"rows={report['modeling_rows']} events={report['events']} "
        f"selected={report['selected_by_brier']} "
        f"baseline_brier={pooled['calendar_month_baseline']['brier']} "
        f"ridge_brier={pooled['ridge_logistic']['brier']} "
        f"booster_brier={pooled['gradient_boosted_trees']['brier']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
