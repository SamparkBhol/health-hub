#!/usr/bin/env python3
"""Write the deterministic forecast-software demonstration report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both `python -m scripts.make_synthetic` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.forecasting import build_synthetic_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic/forecast_report.json")
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--horizon-weeks", type=int, choices=(1, 2, 4, 8, 12), default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build_synthetic_report(args.seed, args.horizon_weeks),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
