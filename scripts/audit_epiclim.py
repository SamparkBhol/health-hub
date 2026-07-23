#!/usr/bin/env python3
"""Reproduce the public EpiClim suitability audit without pandas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

URL = "https://zenodo.org/api/records/14580510/files/Final_data.csv/content"
EXPECTED_SHA256 = "7348076420202f8146ec2d36f36423cebd31af3cfbb8784e8c01e84b8ce0fb31"
EXPECTED_MD5 = "a6c961b95a454226e4720ae1745f9f16"  # upstream Zenodo checksum
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024


def digest(path: Path, name: str) -> str:
    value = hashlib.new(name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(target: Path) -> None:
    request = urllib.request.Request(  # noqa: S310 - URL is a pinned HTTPS constant
        URL, headers={"User-Agent": "OdishaHealthHub-Audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise SystemExit("EpiClim response exceeds the 8 MiB audit limit")
        total = 0
        with target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise SystemExit("EpiClim response exceeds the 8 MiB audit limit")
                handle.write(chunk)


def ordinal_week(value: str) -> int | None:
    match = re.search(r"(\d+)", value or "")
    return int(match.group(1)) if match else None


def iso_week_distance(value: date, labelled_week: int) -> int | None:
    """Return the nearest ISO-week distance, including New-Year wraparound."""

    observed = value.isocalendar()
    observed_monday = date.fromisocalendar(observed.year, observed.week, 1)
    distances: list[int] = []
    for iso_year in (observed.year - 1, observed.year, observed.year + 1):
        try:
            candidate = date.fromisocalendar(iso_year, labelled_week, 1)
        except ValueError:
            continue
        distances.append(abs((candidate - observed_monday).days) // 7)
    return min(distances) if distances else None


def audit(path: Path) -> dict[str, object]:
    actual_sha = digest(path, "sha256")
    actual_md5 = digest(path, "md5")  # noqa: S324 - compatibility check, not security
    if actual_sha != EXPECTED_SHA256 or actual_md5 != EXPECTED_MD5:
        raise SystemExit(
            f"dataset hash mismatch: sha256={actual_sha} md5={actual_md5}"
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    odisha = [row for row in rows if row["state_ut"].strip().casefold() == "odisha"]
    disease_counts = Counter(row["Disease"].strip() for row in odisha)
    years = [int(row["year"]) for row in rows]
    year_counts = Counter(years)
    deaths_blank = sum(not row["Deaths"].strip() for row in rows)
    numeric_cases: list[float] = []
    non_numeric_cases: list[str] = []
    for row in rows:
        try:
            numeric_cases.append(float(row["Cases"]))
        except ValueError:
            non_numeric_cases.append(row["Cases"])

    mismatch = 0
    comparable = 0
    for row in rows:
        expected_week = ordinal_week(row["week_of_outbreak"])
        if expected_week is None:
            continue
        try:
            observed_date = date(int(row["year"]), int(row["mon"]), int(row["day"]))
        except ValueError:
            continue
        distance = iso_week_distance(observed_date, expected_week)
        if distance is None:
            continue
        comparable += 1
        if distance > 1:
            mismatch += 1

    district_coordinates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in odisha:
        district_coordinates[row["district"].strip()].add(
            (row["Latitude"].strip(), row["Longitude"].strip())
        )

    def last_year(disease: str) -> int | None:
        values = [int(row["year"]) for row in odisha if row["Disease"].strip() == disease]
        return max(values) if values else None

    return {
        "schema_version": "1.0.0",
        "audit_date": "2026-07-21",
        "source": {
            "record": "https://zenodo.org/records/14580510",
            "download": URL,
            "sha256": actual_sha,
            "md5": actual_md5,
            "authority_status": "secondary_derived",
        },
        "national": {
            "rows": len(rows),
            "year_min": min(years),
            "year_max": max(years),
            "year_counts_selected": {
                str(year): year_counts[year] for year in (2016, 2020, 2021, 2022)
            },
            "numeric_case_zero_rows": sum(value == 0 for value in numeric_cases),
            "numeric_case_min": min(numeric_cases),
            "non_numeric_case_values": non_numeric_cases,
            "deaths_blank_rows": deaths_blank,
            "deaths_blank_fraction": round(deaths_blank / len(rows), 6),
            "week_index_comparable_rows": comparable,
            "week_index_mismatch_gt_one_week_rows": mismatch,
            "week_index_mismatch_gt_one_week_fraction": round(mismatch / comparable, 6),
            "week_distance_convention": (
                "nearest valid ISO week across the observed ISO year and adjacent years"
            ),
        },
        "odisha": {
            "rows": len(odisha),
            "rows_per_district_year": round(len(odisha) / (30 * 14), 6),
            "disease_counts": dict(sorted(disease_counts.items())),
            "distinct_district_strings": len({row["district"].strip() for row in odisha}),
            "aes_rows": disease_counts.get("Acute Encephalitis Syndrome", 0),
            "scrub_typhus_rows": sum(
                "scrub" in row["Disease"].casefold() for row in odisha
            ),
            "leptospirosis_rows": sum(
                "leptosp" in row["Disease"].casefold() for row in odisha
            ),
            "malaria_last_year": last_year("Malaria"),
            "dengue_last_year": last_year("Dengue"),
            "district_coordinate_sets_all_singleton": all(
                len(values) <= 1 for values in district_coordinates.values()
            ),
        },
        "eligibility": {
            "district_week_count_forecast": "ineligible",
            "reason_codes": [
                "POSITIVE_ONLY_EVENT_CATALOGUE",
                "NO_PUBLISHED_DISTRICT_WEEK_NIL_PANEL",
                "NO_EXPECTED_REPORT_DENOMINATOR",
                "TEMPORAL_INDEX_INCONSISTENCY",
                "STALE_DISEASE_SPECIFIC_ODISHA_HISTORY",
            ],
            "missing_rows_are_zero": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/epiclim/audit.json"))
    parser.add_argument(
        "--save-dataset",
        type=Path,
        default=None,
        help=(
            "Also write the hash-verified CSV here so the modelling target is "
            "reproducible offline. The file stays a positive-only event catalogue."
        ),
    )
    args = parser.parse_args()
    if args.input:
        result = audit(args.input)
        verified = args.input
        if args.save_dataset:
            args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
            args.save_dataset.write_bytes(verified.read_bytes())
    else:
        with tempfile.TemporaryDirectory(prefix="epiclim-audit-") as directory:
            source = Path(directory) / "Final_data.csv"
            fetch(source)
            result = audit(source)
            if args.save_dataset:
                args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
                args.save_dataset.write_bytes(source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
