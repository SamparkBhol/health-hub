"""Validated boundary for authorised Odisha routine-surveillance aggregates.

Public articles and the positive-only IDSP/EpiClim catalogue are useful evidence
inputs, but neither is a district-week disease panel.  This module is the only
supported entry point for the data that can power an operational outbreak model:
one aggregate row per district, disease, epidemiological week and *knowledge
vintage*.  It deliberately rejects patient data and it never fills a missing
week with zero.

The module is dependency-free so an authorised State export can be audited on a
department laptop before it is ever copied into an application database.  It is
also deliberately separate from the public-source crawler: official counts may
not be inferred from media coverage.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .target import load_alias_index, week_start

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_PATH = REPO_ROOT / "data" / "authorised_surveillance" / "district_week.csv"
TEMPLATE_PATH = REPO_ROOT / "data" / "templates" / "authorised_surveillance_district_week.csv"

# A three-year window is the minimum useful starting point for seasonal
# district-week calibration. It is intentionally a data-readiness floor, not a
# promise that three seasons are statistically sufficient for every disease.
MINIMUM_COMPLETE_WEEKS_PER_DISTRICT_DISEASE = 156
MINIMUM_CASE_VOLUME_COMPLETENESS = 0.80
REQUIRED_COLUMNS = (
    "district_id",
    "disease",
    "week_start",
    "cases",
    "population",
    "reporting_units_expected",
    "reporting_units_received",
    "case_volume_completeness",
    "known_at",
    "case_definition_version",
    "outbreak_threshold_per_100k",
    "threshold_version",
    "source_vintage",
)
PRIVACY_BOUNDARY = (
    "aggregate district/week/disease rows only; no persons, addresses, phones, "
    "line lists or free text"
)
VINTAGE_RULE = (
    "latest known_at value is selected per district/disease/week; older values remain "
    "available for reporting-delay analysis"
)


class SurveillanceContractError(ValueError):
    """The aggregate export is not suitable for model training."""


@dataclass(frozen=True, slots=True)
class SurveillanceObservation:
    district_id: str
    disease: str
    week_start: date
    cases: int
    population: int
    reporting_units_expected: int
    reporting_units_received: int
    case_volume_completeness: float
    known_at: date
    case_definition_version: str
    outbreak_threshold_per_100k: float
    threshold_version: str
    source_vintage: str

    @property
    def rate_per_100k(self) -> float:
        return self.cases * 100_000.0 / self.population

    @property
    def reporting_unit_completeness(self) -> float:
        return self.reporting_units_received / self.reporting_units_expected


def _parse_date(value: str, *, field: str, row_number: int) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise SurveillanceContractError(
            f"row {row_number}: {field} must be an ISO date (YYYY-MM-DD)"
        ) from exc


def _parse_int(value: str, *, field: str, row_number: int, minimum: int = 0) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise SurveillanceContractError(f"row {row_number}: {field} must be an integer") from exc
    if parsed < minimum:
        raise SurveillanceContractError(f"row {row_number}: {field} must be >= {minimum}")
    return parsed


def _parse_fraction(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise SurveillanceContractError(
            f"row {row_number}: {field} must be a decimal in [0, 1]"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise SurveillanceContractError(f"row {row_number}: {field} must be in [0, 1]")
    return parsed


def _parse_fraction_or_positive(value: str, *, field: str, row_number: int) -> float:
    """Parse a non-negative rate threshold; zero supports elimination targets."""

    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise SurveillanceContractError(
            f"row {row_number}: {field} must be a non-negative decimal"
        ) from exc
    if parsed < 0.0:
        raise SurveillanceContractError(f"row {row_number}: {field} must be >= 0")
    return parsed


def _canonical_district(value: str, aliases: dict[str, str], row_number: int) -> str:
    raw = value.strip()
    if raw.startswith("OD-DIST-") and raw in aliases.values():
        return raw
    normalised = "".join(character for character in raw.casefold() if character.isalnum())
    district_id = aliases.get(normalised)
    if district_id is None:
        raise SurveillanceContractError(
            f"row {row_number}: district_id {raw!r} is not an Odisha district alias"
        )
    return district_id


def load_export(path: Path | str = DEFAULT_EXPORT_PATH) -> tuple[SurveillanceObservation, ...]:
    """Read a no-PII aggregate export and fail closed on all ambiguous rows."""

    source = Path(path)
    if not source.exists():
        raise SurveillanceContractError(
            f"authorised aggregate export is missing at {source}; use template {TEMPLATE_PATH}"
        )
    aliases = load_alias_index()
    allowed_diseases: set[str] = set()
    # Import locally to keep the export contract usable without the crawler.
    from workers.ingestion.diseases import DiseaseLexicon

    allowed_diseases.update(DiseaseLexicon.load().terms)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_COLUMNS if field not in actual]
        extra = [field for field in actual if field not in REQUIRED_COLUMNS]
        if missing or extra:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            raise SurveillanceContractError(
                "export columns do not match contract: " + "; ".join(details)
            )

        output: list[SurveillanceObservation] = []
        identities: set[tuple[str, str, date, date]] = set()
        for row_number, row in enumerate(reader, start=2):
            district_id = _canonical_district(row["district_id"], aliases, row_number)
            disease = row["disease"].strip()
            if disease not in allowed_diseases:
                raise SurveillanceContractError(
                    f"row {row_number}: disease {disease!r} is not in the versioned disease lexicon"
                )
            observed_week = _parse_date(
                row["week_start"], field="week_start", row_number=row_number
            )
            if observed_week != week_start(observed_week):
                raise SurveillanceContractError(
                    f"row {row_number}: week_start must be an ISO Monday, not {observed_week}"
                )
            known_at = _parse_date(row["known_at"], field="known_at", row_number=row_number)
            if known_at < observed_week:
                raise SurveillanceContractError(
                    f"row {row_number}: known_at predates its epidemiological week"
                )
            cases = _parse_int(row["cases"], field="cases", row_number=row_number)
            population = _parse_int(
                row["population"], field="population", row_number=row_number, minimum=1
            )
            expected = _parse_int(
                row["reporting_units_expected"],
                field="reporting_units_expected",
                row_number=row_number,
                minimum=1,
            )
            received = _parse_int(
                row["reporting_units_received"],
                field="reporting_units_received",
                row_number=row_number,
            )
            if received > expected:
                raise SurveillanceContractError(
                    f"row {row_number}: reporting_units_received exceeds expected"
                )
            completeness = _parse_fraction(
                row["case_volume_completeness"],
                field="case_volume_completeness",
                row_number=row_number,
            )
            definition = row["case_definition_version"].strip()
            threshold = _parse_fraction_or_positive(
                row["outbreak_threshold_per_100k"],
                field="outbreak_threshold_per_100k",
                row_number=row_number,
            )
            threshold_version = row["threshold_version"].strip()
            vintage = row["source_vintage"].strip()
            if not definition or not threshold_version or not vintage:
                raise SurveillanceContractError(
                    f"row {row_number}: case_definition_version, threshold_version and "
                    "source_vintage are required"
                )
            identity = (district_id, disease, observed_week, known_at)
            if identity in identities:
                raise SurveillanceContractError(
                    f"row {row_number}: duplicate district/disease/week/known_at observation"
                )
            identities.add(identity)
            output.append(
                SurveillanceObservation(
                    district_id=district_id,
                    disease=disease,
                    week_start=observed_week,
                    cases=cases,
                    population=population,
                    reporting_units_expected=expected,
                    reporting_units_received=received,
                    case_volume_completeness=completeness,
                    known_at=known_at,
                    case_definition_version=definition,
                    outbreak_threshold_per_100k=threshold,
                    threshold_version=threshold_version,
                    source_vintage=vintage,
                )
            )
    if not output:
        raise SurveillanceContractError("authorised aggregate export contains no rows")
    return tuple(
        sorted(
            output,
            key=lambda item: (item.disease, item.district_id, item.week_start, item.known_at),
        )
    )


def _latest_vintage(
    rows: tuple[SurveillanceObservation, ...],
) -> tuple[SurveillanceObservation, ...]:
    """Choose the latest known value per epidemiological observation explicitly."""

    selected: dict[tuple[str, str, date], SurveillanceObservation] = {}
    for row in rows:
        key = (row.district_id, row.disease, row.week_start)
        prior = selected.get(key)
        if prior is None or row.known_at > prior.known_at:
            selected[key] = row
    return tuple(
        sorted(
            selected.values(), key=lambda item: (item.disease, item.district_id, item.week_start)
        )
    )


def audit_export(path: Path | str = DEFAULT_EXPORT_PATH) -> dict[str, Any]:
    """Return a machine-readable training gate without fitting a model."""

    source = Path(path)
    if not source.exists():
        return {
            "status": "awaiting_authorised_aggregate_export",
            "eligible_for_training": False,
            "path": str(source),
            "template_path": str(TEMPLATE_PATH),
            "reason_codes": ["AUTHORISED_DISTRICT_WEEK_EXPORT_NOT_PRESENT"],
            "required_columns": list(REQUIRED_COLUMNS),
            "minimum_complete_weeks_per_district_disease": (
                MINIMUM_COMPLETE_WEEKS_PER_DISTRICT_DISEASE
            ),
            "minimum_case_volume_completeness": MINIMUM_CASE_VOLUME_COMPLETENESS,
            "privacy_boundary": PRIVACY_BOUNDARY,
        }
    try:
        rows = load_export(source)
    except SurveillanceContractError as exc:
        return {
            "status": "rejected_contract",
            "eligible_for_training": False,
            "path": str(source),
            "reason_codes": ["AUTHORISED_EXPORT_CONTRACT_INVALID"],
            "detail": str(exc),
            "required_columns": list(REQUIRED_COLUMNS),
        }

    latest = _latest_vintage(rows)
    raw = source.read_bytes()
    grouped: dict[tuple[str, str], list[SurveillanceObservation]] = defaultdict(list)
    for row in latest:
        grouped[(row.disease, row.district_id)].append(row)
    diseases: dict[str, dict[str, Any]] = {}
    for disease in sorted({row.disease for row in latest}):
        cells = [rows for (item_disease, _), rows in grouped.items() if item_disease == disease]

        def is_contiguous(cell: list[SurveillanceObservation]) -> bool:
            weeks = sorted(item.week_start for item in cell)
            if not weeks:
                return False
            expected = ((weeks[-1] - weeks[0]).days // 7) + 1
            return len(weeks) == expected

        complete_cells = [
            cell
            for cell in cells
            if len(cell) >= MINIMUM_COMPLETE_WEEKS_PER_DISTRICT_DISEASE
            and is_contiguous(cell)
            and all(
                item.case_volume_completeness >= MINIMUM_CASE_VOLUME_COMPLETENESS for item in cell
            )
        ]
        observed_weeks = sorted(item.week_start for cell in cells for item in cell)
        diseases[disease] = {
            "districts_observed": len(cells),
            "districts_meeting_history_and_completeness_floor": len(complete_cells),
            "districts_with_explicit_nil_or_case_row_every_week": sum(
                int(is_contiguous(cell)) for cell in cells
            ),
            "rows_latest_vintage": sum(len(cell) for cell in cells),
            "week_range": (
                {"from": observed_weeks[0].isoformat(), "to": observed_weeks[-1].isoformat()}
                if observed_weeks
                else None
            ),
            "eligible": len(complete_cells) >= 5,
        }
    eligible = bool(diseases) and all(item["eligible"] for item in diseases.values())
    reasons = [] if eligible else ["INSUFFICIENT_COMPLETE_DISTRICT_DISEASE_HISTORY"]
    return {
        "status": "eligible_for_model_training" if eligible else "insufficient_evidence",
        "eligible_for_training": eligible,
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows_all_vintages": len(rows),
        "rows_latest_vintage": len(latest),
        "diseases": diseases,
        "reason_codes": reasons,
        "minimum_complete_weeks_per_district_disease": MINIMUM_COMPLETE_WEEKS_PER_DISTRICT_DISEASE,
        "minimum_case_volume_completeness": MINIMUM_CASE_VOLUME_COMPLETENESS,
        "vintage_rule": VINTAGE_RULE,
        "privacy_boundary": PRIVACY_BOUNDARY,
    }
