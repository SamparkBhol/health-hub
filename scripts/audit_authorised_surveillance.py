"""Audit a no-PII district-week surveillance export before forecast training.

Usage:
    uv run python scripts/audit_authorised_surveillance.py path/to/export.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main() -> int:
    from packages.forecasting.authorised_surveillance import (  # noqa: PLC0415
        DEFAULT_EXPORT_PATH,
        audit_export,
    )

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPORT_PATH
    print(json.dumps(audit_export(path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
