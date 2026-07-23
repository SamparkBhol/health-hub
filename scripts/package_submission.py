#!/usr/bin/env python3
"""Build a clean source archive from the current working tree."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "odisha-health-hub.zip"

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".playwright-mcp",
    ".pytest_cache",
    ".remember",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "runtime",
}
EXCLUDED_NAMES = {
    ".env",
    ".env.production",
    "dg.html",
    "blank-2018.png",
    "malaria-map.png",
    "ncdc.html",
    "spn.html",
    DESTINATION.name,
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts and relative.parts[0] == "models":
        return False
    if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative.parts[:2] in {("data", "raw"), ("data", "quarantine")}:
        return False
    if relative.parts[:2] == ("apps", "web") and "dist" in relative.parts:
        return False
    return path.is_file()


def main() -> int:
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    temporary = DESTINATION.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    os.replace(temporary, DESTINATION)
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    print(f"{DESTINATION}")
    print(f"files={len(files)} bytes={DESTINATION.stat().st_size} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
