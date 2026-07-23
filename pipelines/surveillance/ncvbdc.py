"""NCVBDC district-wise annual malaria tables for Odisha.

The National Centre for Vector Borne Diseases Control publishes one annual PDF
containing district tables for every state.  These are official annual malaria
statistics, not a current weekly surveillance feed.  This module downloads the
reports, verifies that the complete 30-district Odisha table was recovered, and
writes a small reproducible CSV plus a hash manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "public_health"
CSV_PATH = DATA_ROOT / "ncvbdc_malaria_annual.csv"
MANIFEST_PATH = DATA_ROOT / "ncvbdc_manifest.json"
GAZETTEER_PATH = PROJECT_ROOT / "data" / "gazetteer" / "odisha_district_aliases.csv"
NCVBDC_ORIGIN = "https://ncvbdc.mohfw.gov.in"


class NCVBDCError(RuntimeError):
    """The annual report could not be retrieved or validated."""


@dataclass(frozen=True, slots=True)
class MalariaAnnualRow:
    year: int
    district_id: str
    district_name: str
    population_thousands: float | None
    total_cases: int | None
    api: float
    aber: float
    spr: float
    pf_percent: float
    deaths: int | None
    source_url: str
    source_sha256: str
    metric_scope: str = "official_annual_malaria_report"


def annual_report_url(year: int) -> str:
    if not 2010 <= year <= 2024:
        raise ValueError("NCVBDC collector supports reports from 2010 through 2024")
    if year <= 2016:
        name = f"Malaria-Annual-Report-{year}.pdf"
    elif year in {2017, 2018}:
        name = f"Annual-Report-{year}.pdf"
    elif year <= 2023:
        name = f"Malaria-AnnualReport-{year}.pdf"
    else:
        name = "Malaria-Annual-Report-2024.pdf"
    return f"{NCVBDC_ORIGIN}/Doc/{name}"


def _gazetteer() -> list[dict[str, Any]]:
    with GAZETTEER_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise NCVBDCError("Odisha district gazetteer must contain exactly 30 districts")
    return rows


def _patterns() -> list[tuple[dict[str, Any], re.Pattern[str]]]:
    output = []
    for row in _gazetteer():
        aliases = set(str(row["aliases"]).split("|")) | {str(row["canonical_name"])}
        english = sorted((value for value in aliases if value.isascii()), key=len, reverse=True)
        expression = (
            r"^(?:\d+\s+)?(?:Odisha\s+)?("
            + "|".join(re.escape(value) for value in english)
            + r")(?=\s|\(|$)"
        )
        output.append((row, re.compile(expression, re.IGNORECASE)))
    return output


_NUMBER = re.compile(r"(?<![A-Za-z])(?:#{4,}|-?\d+(?:\.\d+)?)")


def _numbers(value: str) -> list[float | None]:
    return [None if token.startswith("#") else float(token) for token in _NUMBER.findall(value)]


def _pdf_lines(body: bytes) -> tuple[list[str], str]:
    document = pdfium.PdfDocument(body)
    page_texts: list[str] = []
    lines: list[str] = []
    for page in document:
        text = page.get_textpage().get_text_bounded()
        page_texts.append(text)
        lines.extend(" ".join(line.split()) for line in text.replace("\r", "\n").split("\n"))
    return [line for line in lines if line], " ".join(" ".join(text.split()) for text in page_texts)


def _split_rows(year: int, body: bytes) -> dict[str, tuple[dict[str, Any], list[float | None]]]:
    lines, flattened = _pdf_lines(body)
    patterns = _patterns()
    found: dict[str, tuple[dict[str, Any], list[float | None]]] = {}
    for line in lines:
        for row, pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            values = _numbers(line[match.end() :])
            # 2018 has page-layout line breaks inside the last eight rows.  It
            # is handled below from the whitespace-normalised table.
            if len(values) >= (8 if year in {2017, 2018} else 12):
                found[str(row["district_id"])] = (row, values)
            break

    if year == 2018 and len(found) < 30:
        start = flattened.find("1 Angul ")
        end = flattened.find("State Total", start)
        table = flattened[start:end]
        for row, _pattern in patterns:
            aliases = sorted(
                {
                    value
                    for value in str(row["aliases"]).split("|")
                    if value.isascii()
                }
                | {str(row["canonical_name"])},
                key=len,
                reverse=True,
            )
            name = "|".join(re.escape(value) for value in aliases)
            match = re.search(
                rf"(?:^|\s)\d+\s+(?:{name})(?:\s*\([^)]*\))?\s+"
                rf"(?P<values>(?:#{{4,}}|-?\d+(?:\.\d+)?)(?:\s+(?:#{{4,}}|-?\d+(?:\.\d+)?)){{7}})",
                table,
                re.IGNORECASE,
            )
            if match:
                found[str(row["district_id"])] = (row, _numbers(match.group("values")))

    if len(found) != 30:
        missing = [
            str(row["canonical_name"])
            for row in _gazetteer()
            if str(row["district_id"]) not in found
        ]
        raise NCVBDCError(
            f"{year} Odisha table yielded {len(found)} districts; missing {', '.join(missing)}"
        )
    return found


def parse_annual_report(year: int, body: bytes, *, source_url: str) -> list[MalariaAnnualRow]:
    digest = hashlib.sha256(body).hexdigest()
    parsed = _split_rows(year, body)
    output: list[MalariaAnnualRow] = []
    for gazetteer_row in _gazetteer():
        district_id = str(gazetteer_row["district_id"])
        _, values = parsed[district_id]
        if year <= 2016:
            population, total, pf, aber, api, spr = (
                values[0], values[4], values[5], values[6], values[7], values[8]
            )
            deaths = values[11]
        elif year in {2017, 2018}:
            population = None
            total = None
            pf, aber, api, spr = values[0], values[1], values[2], values[3]
            deaths = sum(value or 0 for value in values[6:8])
        else:
            population = values[0]
            total = values[-10]
            pf, aber, api, spr = values[-9], values[-8], values[-7], values[-6]
            deaths = sum(value or 0 for value in values[-2:])
        if pf is None or aber is None or api is None or spr is None:
            raise NCVBDCError(f"{year} {district_id} contains a missing required metric")
        output.append(
            MalariaAnnualRow(
                year=year,
                district_id=district_id,
                district_name=str(gazetteer_row["canonical_name"]),
                population_thousands=None if population is None else float(population),
                total_cases=None if total is None else int(total),
                api=float(api),
                aber=float(aber),
                spr=float(spr),
                pf_percent=float(pf),
                deaths=None if deaths is None else int(deaths),
                source_url=source_url,
                source_sha256=digest,
            )
        )
    return output


def _download(url: str, timeout: int = 90) -> bytes:
    if not url.startswith(f"{NCVBDC_ORIGIN}/Doc/"):
        raise NCVBDCError("refusing a report URL outside the pinned NCVBDC origin")
    request = Request(  # noqa: S310 - URL is constrained to the HTTPS origin above
        url, headers={"User-Agent": "OdishaHealthHub/1.0 (+public-data-audit)"}
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS origin
        body = response.read()
    if not body.startswith(b"%PDF"):
        raise NCVBDCError(f"{url} did not return a PDF")
    return body


def collect_ncvbdc_annual(
    *, years: range = range(2010, 2025), destination: Path = DATA_ROOT
) -> list[MalariaAnnualRow]:
    destination.mkdir(parents=True, exist_ok=True)
    all_rows: list[MalariaAnnualRow] = []
    reports: list[dict[str, Any]] = []
    for year in years:
        url = annual_report_url(year)
        body = _download(url)
        rows = parse_annual_report(year, body, source_url=url)
        all_rows.extend(rows)
        reports.append(
            {
                "year": year,
                "source_url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "byte_length": len(body),
                "district_rows": len(rows),
            }
        )
    csv_path = destination / CSV_PATH.name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(all_rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)
    manifest = {
        "schema_version": "1.0.0",
        "retrieved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "publisher": "National Centre for Vector Borne Diseases Control, MoHFW",
        "scope": "Official annual district-wise malaria reports; not weekly incidence",
        "row_count": len(all_rows),
        "district_count": len({row.district_id for row in all_rows}),
        "reports": reports,
        "derived_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (destination / MANIFEST_PATH.name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return all_rows


def load_ncvbdc_rows(path: Path = CSV_PATH) -> list[MalariaAnnualRow]:
    if not path.is_file():
        return []
    output: list[MalariaAnnualRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(
                MalariaAnnualRow(
                    year=int(row["year"]),
                    district_id=row["district_id"],
                    district_name=row["district_name"],
                    population_thousands=(
                        None
                        if not row["population_thousands"]
                        else float(row["population_thousands"])
                    ),
                    total_cases=None if not row["total_cases"] else int(row["total_cases"]),
                    api=float(row["api"]),
                    aber=float(row["aber"]),
                    spr=float(row["spr"]),
                    pf_percent=float(row["pf_percent"]),
                    deaths=None if not row["deaths"] else int(row["deaths"]),
                    source_url=row["source_url"],
                    source_sha256=row["source_sha256"],
                    metric_scope=row["metric_scope"],
                )
            )
    return output
