"""Build the offline authorised aggregate outbreak-model artefact.

Usage:
    uv run python scripts/train_authorised_forecast.py \
      data/authorised_surveillance/district_week.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.forecasting.operational import ARTEFACT_PATH, train_and_write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Authorised no-PII district-week CSV")
    parser.add_argument("--output", type=Path, default=ARTEFACT_PATH)
    arguments = parser.parse_args()
    report = train_and_write(path=arguments.export, output=arguments.output)
    print(json.dumps({"status": report["status"], "output": str(arguments.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
