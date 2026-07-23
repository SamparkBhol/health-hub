#!/usr/bin/env python3
"""Fail fast when the reproducible demo prerequisites are not satisfied."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Validate API/runtime prerequisites without requiring Node.js.",
    )
    args = parser.parse_args()
    failures: list[str] = []
    print(f"python={sys.version.split()[0]}")
    commands = (
        ("tesseract", "pdftoppm", "pdftotext")
        if args.runtime
        else ("node", "npm", "tesseract", "pdftoppm", "pdftotext")
    )
    for command in commands:
        location = shutil.which(command)
        print(f"{command}={location or 'missing'}")
        if not location:
            failures.append(f"missing command: {command}")

    if shutil.which("tesseract"):
        tesseract = shutil.which("tesseract")
        assert tesseract is not None
        result = subprocess.run(  # noqa: S603 - resolved executable; constant arguments
            [tesseract, "--list-langs"], capture_output=True, text=True, check=False
        )
        languages = {line.strip() for line in result.stdout.splitlines()[1:]}
        print(f"tesseract_languages={','.join(sorted(languages))}")
        missing = {"eng", "hin", "ori"} - languages
        if missing:
            failures.append(f"missing Tesseract language packs: {sorted(missing)}")

    source_path = ROOT / "config" / "sources.yaml"
    try:
        registry = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        sources = registry["sources"]
        identifiers = [item["id"] for item in sources]
        # Assert the properties that matter rather than an exact count, so adding a
        # verified source is not a build break. What must hold is that identifiers
        # are unique and that every one of the three assignment languages actually
        # has enabled live routes -- a registry that quietly lost its Odia or Hindi
        # coverage would still "work" but would no longer be the product.
        if len(identifiers) != len(set(identifiers)):
            failures.append("source registry contains duplicate route identifiers")
        enabled = [item for item in sources if item["enabled"]]
        if len(enabled) < 20:
            failures.append(
                f"source registry must keep at least 20 enabled routes, found {len(enabled)}"
            )
        for language in ("en", "hi", "or"):
            if not any(language in item["languages"] for item in enabled):
                failures.append(f"no enabled source route covers language {language!r}")
        print(f"registered_sources={len(identifiers)}")
    except Exception as exc:  # noqa: BLE001 - diagnostic must aggregate failures
        failures.append(f"source registry invalid: {exc}")

    boundary_directory = ROOT / "data" / "boundaries"
    try:
        manifest = json.loads((boundary_directory / "manifest.json").read_text())
        asset = boundary_directory / manifest["asset"]
        if sha256(asset) != manifest["asset_sha256"]:
            failures.append("district boundary hash mismatch")
        features = json.loads(asset.read_text())["features"]
        if len(features) != 30:
            failures.append(f"district boundary count is {len(features)}, not 30")
        print(f"boundary_sha256={manifest['asset_sha256']}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"boundary asset invalid: {exc}")

    audit = ROOT / "data" / "epiclim" / "audit.json"
    if not audit.exists():
        failures.append("missing reproducible EpiClim audit")
    else:
        print(f"epiclim_audit_sha256={sha256(audit)}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("doctor=ok")


if __name__ == "__main__":
    main()
