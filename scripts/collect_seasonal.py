#!/usr/bin/env python3
"""Refresh the 30-district, three-month environmental ensemble outlook."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.environmental.seasonal import refresh_seasonal_outlook


def main() -> int:
    payload = refresh_seasonal_outlook()
    print(
        f"Seasonal outlook: {len(payload['districts'])} districts, "
        f"{payload['ensemble_members']} ensemble members"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
