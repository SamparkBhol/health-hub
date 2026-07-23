"""Refresh authoritative bundled public-health data products."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.surveillance import collect_hmis_district_months, collect_ncvbdc_annual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hmis", action="store_true")
    parser.add_argument("--skip-ncvbdc", action="store_true")
    args = parser.parse_args()
    if not args.skip_ncvbdc:
        ncvbdc_rows = collect_ncvbdc_annual()
        print(f"NCVBDC: {len(ncvbdc_rows)} validated annual district rows")
    if not args.skip_hmis:
        hmis_rows = collect_hmis_district_months()
        print(f"HMIS: {len(hmis_rows)} validated monthly district rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
