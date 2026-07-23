"""OGD Odisha monthly HMIS district indicators.

The Open Government Data Platform exposes one CSV per month from FY 2012-13
through 2019-20.  Values used here are facility-reported test/service records;
they are not deduplicated patient counts and must never be labelled incidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from .ncvbdc import DATA_ROOT, GAZETTEER_PATH

CSV_PATH = DATA_ROOT / "hmis_district_monthly.csv"
MANIFEST_PATH = DATA_ROOT / "hmis_manifest.json"
RESOURCE_TEMPLATE = (
    "https://www.data.gov.in/resource/"
    "item-wise-hmis-report-district-level-odisha-{month}-{fiscal_year}"
)
FILE_TEMPLATE = (
    "https://www.data.gov.in/sites/default/files/dataurl21122020/"
    "hmis-item-{fiscal_year}-mn-od-for-{month_short}.csv"
)
MONTHS = (
    ("april", "Apr", 4), ("may", "May", 5), ("june", "Jun", 6),
    ("july", "Jul", 7), ("august", "Aug", 8), ("september", "Sep", 9),
    ("october", "Oct", 10), ("november", "Nov", 11), ("december", "Dec", 12),
    ("january", "Jan", 1), ("february", "Feb", 2), ("march", "Mar", 3),
)

PARAMETERS = {
    "blood_smears": (
        "total blood smears examined for malaria",
        "number of blood smears examined for malaria",
    ),
    "microscopy_pv": (
        "malaria (microscopy tests ) - plasmodium vivax test positive",
        "out of blood smears examined for malaria, number of blood smears tested "
        "positive for plasmodium vivax",
    ),
    "microscopy_pf": (
        "malaria (microscopy tests ) - plasmodium falciparum test positive",
        "out of blood smears examined for malaria, number of blood smears tested "
        "positive for plasmodium falciparum",
    ),
    "rdt_tests": ("rdt conducted for malaria",),
    "rdt_pv": ("malaria (rdt) - plasmodium vivax test positive",),
    "rdt_pf": ("malaria (rdt) - plamodium falciparum test positive",),
    "dengue_rdt": ("dengue - rdt test positive",),
    "dengue_elisa": (
        "dengue - enzyme- linked immuno sorbent assay (elisa) test positive",
    ),
    "child_diarrhoea": (
        "childhood diseases - diarrhoea",
        "number of cases of diarrhoea and dehydration reported in children below "
        "5 years of age",
    ),
}
REQUIRED_PARAMETERS = {"blood_smears", "microscopy_pv", "microscopy_pf", "child_diarrhoea"}
FISCAL_YEARS = tuple(
    f"{start_year}-{str(start_year + 1)[-2:]}" for start_year in range(2012, 2020)
)


class HMISError(RuntimeError):
    """A public HMIS resource did not satisfy its data contract."""


@dataclass(frozen=True, slots=True)
class HMISRow:
    fiscal_year: str
    month: int
    period_start: str
    district_id: str
    district_name: str
    malaria_microscopy_tests: int | None
    malaria_microscopy_positive_records: int | None
    malaria_microscopy_positivity: float | None
    malaria_rdt_positive_records: int | None
    malaria_positive_records: int | None
    malaria_tests: int | None
    malaria_test_positivity: float | None
    dengue_positive_records: int | None
    childhood_diarrhoea_records: int | None
    source_url: str
    resource_url: str
    source_sha256: str
    metric_scope: str = "facility_reported_test_and_service_records"


def _normalise(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").strip().lower().split())


def _number(value: str) -> int | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"na", "n/a", "null", "-"}:
        return None
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return None


def _district_lookup() -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    with GAZETTEER_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = set(row["aliases"].split("|")) | {row["canonical_name"]}
            for value in values:
                if value.isascii():
                    output[_normalise(value)] = (row["district_id"], row["canonical_name"])
    return output


def _period_start(fiscal_year: str, month: int) -> date:
    first = int(fiscal_year.split("-")[0])
    year = first if month >= 4 else first + 1
    return date(year, month, 1)


def parse_hmis_month(
    body: bytes,
    *,
    fiscal_year: str,
    month: int,
    source_url: str,
    resource_url: str,
) -> list[HMISRow]:
    text = body.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) < 34:
        raise HMISError(f"{source_url} has an unexpected HMIS shape")
    header = rows[0]
    district_lookup = _district_lookup()
    columns: dict[str, tuple[int, str, str]] = {}
    pattern = re.compile(r"^District - (.+?)(?: - Total \[|$)")
    for index, value in enumerate(header):
        match = pattern.match(value.strip())
        if not match:
            continue
        raw_name = match.group(1).lstrip("_")
        resolved = district_lookup.get(_normalise(raw_name))
        if resolved:
            columns[resolved[0]] = (index, resolved[0], resolved[1])
    if len(columns) != 30:
        raise HMISError(f"{source_url} resolved {len(columns)} of 30 district-total columns")

    parameter_rows: dict[str, list[str]] = {}
    wanted = {
        _normalise(alias): key
        for key, aliases in PARAMETERS.items()
        for alias in aliases
    }
    for row in rows[1:]:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        key = wanted.get(_normalise(row[2]))
        if key:
            parameter_rows[key] = row
    missing = REQUIRED_PARAMETERS - set(parameter_rows)
    if missing:
        raise HMISError(f"{source_url} lacks required indicators: {', '.join(sorted(missing))}")

    digest = hashlib.sha256(body).hexdigest()
    output: list[HMISRow] = []
    for _, (column, district_id, district_name) in sorted(columns.items()):
        metrics = {
            key: _number(parameter_rows[key][column]) if key in parameter_rows else None
            for key in PARAMETERS
        }

        microscopy_parts = [metrics["microscopy_pv"], metrics["microscopy_pf"]]
        microscopy_positive = (
            None
            if any(value is None for value in microscopy_parts)
            else sum(microscopy_parts)  # type: ignore[arg-type]
        )
        rdt_parts = [metrics["rdt_pv"], metrics["rdt_pf"]]
        rdt_positive = (
            None if any(value is None for value in rdt_parts) else sum(rdt_parts)  # type: ignore[arg-type]
        )
        malaria_positive = microscopy_positive
        if malaria_positive is not None and rdt_positive is not None:
            malaria_positive += rdt_positive
        tests = metrics["blood_smears"]
        if tests is not None and metrics["rdt_tests"] is not None:
            tests += metrics["rdt_tests"]  # type: ignore[operator]
        positivity = None
        if malaria_positive is not None and tests is not None and tests != 0:
            positivity = malaria_positive / tests
        microscopy_tests = metrics["blood_smears"]
        microscopy_positivity = None
        if (
            microscopy_positive is not None
            and microscopy_tests is not None
            and microscopy_tests != 0
        ):
            microscopy_positivity = microscopy_positive / microscopy_tests
        dengue_parts = [metrics["dengue_rdt"], metrics["dengue_elisa"]]
        dengue = (
            None
            if any(value is None for value in dengue_parts)
            else sum(dengue_parts)  # type: ignore[arg-type]
        )
        output.append(
            HMISRow(
                fiscal_year=fiscal_year,
                month=month,
                period_start=_period_start(fiscal_year, month).isoformat(),
                district_id=district_id,
                district_name=district_name,
                malaria_microscopy_tests=microscopy_tests,
                malaria_microscopy_positive_records=microscopy_positive,
                malaria_microscopy_positivity=microscopy_positivity,
                malaria_rdt_positive_records=rdt_positive,
                malaria_positive_records=malaria_positive,
                malaria_tests=tests,
                malaria_test_positivity=positivity,
                dengue_positive_records=dengue,
                childhood_diarrhoea_records=metrics["child_diarrhoea"],
                source_url=source_url,
                resource_url=resource_url,
                source_sha256=digest,
            )
        )
    return output


def _download(url: str, timeout: int = 30) -> bytes:
    if not url.startswith("https://www.data.gov.in/sites/default/files/"):
        raise HMISError("refusing an HMIS URL outside the pinned data.gov.in origin")
    headers = {
        "User-Agent": "OdishaHealthHub/1.0 (+public-data-audit)",
        "Referer": "https://www.data.gov.in/",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(timeout, connect=10),
                follow_redirects=True,
            )
            response.raise_for_status()
            if not response.content.startswith(b"Indicator,"):
                raise HMISError(f"{url} did not return the expected CSV")
            return response.content
        except (httpx.HTTPError, HMISError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise HMISError(f"failed to retrieve {url}: {last_error}")


def collect_hmis_district_months(
    *,
    fiscal_years: tuple[str, ...] = FISCAL_YEARS,
    destination: Path = DATA_ROOT,
) -> list[HMISRow]:
    destination.mkdir(parents=True, exist_ok=True)
    all_rows: list[HMISRow] = []
    resources: list[dict[str, object]] = []
    requests = [
        (fiscal_year, month_name, month_short, month_number)
        for fiscal_year in fiscal_years
        for month_name, month_short, month_number in MONTHS
    ]
    urls = [
        FILE_TEMPLATE.format(fiscal_year=fiscal_year, month_short=month_short)
        for fiscal_year, _month_name, month_short, _month_number in requests
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        bodies = list(pool.map(_download, urls))
    for (fiscal_year, month_name, month_short, month_number), body in zip(
        requests, bodies, strict=True
    ):
        source_url = FILE_TEMPLATE.format(
            fiscal_year=fiscal_year, month_short=month_short
        )
        resource_url = RESOURCE_TEMPLATE.format(
            month=month_name, fiscal_year=fiscal_year
        )
        rows = parse_hmis_month(
            body,
            fiscal_year=fiscal_year,
            month=month_number,
            source_url=source_url,
            resource_url=resource_url,
        )
        all_rows.extend(rows)
        resources.append(
            {
                "fiscal_year": fiscal_year,
                "month": month_number,
                "source_url": source_url,
                "resource_url": resource_url,
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
        "publisher": "Open Government Data Platform India / Odisha HMIS",
        "scope": (
            "Monthly facility-reported test and service records; provisional; "
            "not deduplicated people or population incidence"
        ),
        "row_count": len(all_rows),
        "district_count": len({row.district_id for row in all_rows}),
        "resources": resources,
        "derived_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (destination / MANIFEST_PATH.name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return all_rows


def load_hmis_rows(path: Path = CSV_PATH) -> list[HMISRow]:
    if not path.is_file():
        return []
    output: list[HMISRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(
                HMISRow(
                    fiscal_year=row["fiscal_year"],
                    month=int(row["month"]),
                    period_start=row["period_start"],
                    district_id=row["district_id"],
                    district_name=row["district_name"],
                    malaria_microscopy_tests=(
                        None
                        if not row["malaria_microscopy_tests"]
                        else int(row["malaria_microscopy_tests"])
                    ),
                    malaria_microscopy_positive_records=(
                        None
                        if not row["malaria_microscopy_positive_records"]
                        else int(row["malaria_microscopy_positive_records"])
                    ),
                    malaria_microscopy_positivity=(
                        None
                        if not row["malaria_microscopy_positivity"]
                        else float(row["malaria_microscopy_positivity"])
                    ),
                    malaria_rdt_positive_records=(
                        None
                        if not row["malaria_rdt_positive_records"]
                        else int(row["malaria_rdt_positive_records"])
                    ),
                    malaria_positive_records=(
                        None
                        if not row["malaria_positive_records"]
                        else int(row["malaria_positive_records"])
                    ),
                    malaria_tests=(
                        None if not row["malaria_tests"] else int(row["malaria_tests"])
                    ),
                    malaria_test_positivity=(
                        None if not row["malaria_test_positivity"]
                        else float(row["malaria_test_positivity"])
                    ),
                    dengue_positive_records=(
                        None
                        if not row["dengue_positive_records"]
                        else int(row["dengue_positive_records"])
                    ),
                    childhood_diarrhoea_records=(
                        None
                        if not row["childhood_diarrhoea_records"]
                        else int(row["childhood_diarrhoea_records"])
                    ),
                    source_url=row["source_url"],
                    resource_url=row["resource_url"],
                    source_sha256=row["source_sha256"],
                    metric_scope=row["metric_scope"],
                )
            )
    return output
