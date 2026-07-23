"""Experimental EpiClim catalogue-row target for Odisha.

WHAT THIS IS
------------
EpiClim (Zenodo record 14580510) is an incomplete, positive-only derived
catalogue transcribed from IDSP weekly outbreak bulletins. It contains no NIL
weeks, no denominators, no publication timestamps and no expected-report count.
For a bounded historical experiment we can define only this dataset-membership
label:

    y[district, ISO week] = 1  if this frozen EpiClim file contains at least one
                               matching row dated in that district-week
                             0  if it contains no matching row

``0`` means only "no matching row is present in this file". It does NOT mean no
official report was published and it does NOT mean no disease occurred. EpiClim
recovers only a subset of the source bulletins, so this label cannot estimate
the official reporting process. A model of ``P(y = 1)`` is an experimental
model of EpiClim row occurrence under its unknown selection process. It is NOT
incidence, NOT a case count, NOT an official-publication probability and NOT an
operational disease forecast.

The catalogue's ``week_of_outbreak`` label disagrees with the row date by more
than one ISO week for 28% of national rows, with some much larger discrepancies.
This experiment indexes rows by ``year``/``mon``/``day`` because those columns
can be parsed consistently; that is an event-date convention, not a claim about
when a bulletin was published or available to a forecaster.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "epiclim" / "Final_data.csv"
GAZETTEER_PATH = REPO_ROOT / "data" / "gazetteer" / "odisha_district_aliases.csv"
EXPECTED_SHA256 = "7348076420202f8146ec2d36f36423cebd31af3cfbb8784e8c01e84b8ce0fb31"

TARGET_KIND = "experimental_epiclim_catalogue_row_occurrence"
TARGET_STATEMENT = (
    "Experimental retrospective probability that the frozen EpiClim file contains "
    "at least one row dated in this district-week. EpiClim is incomplete and "
    "positive-only: zero means no matching row in this file, not no official "
    "publication and not no disease. This is not incidence, not a case count, not "
    "an official-report probability and not an operational outbreak forecast."
)

# EpiClim district strings that the shared gazetteer does not carry. Each entry
# is an explicit, reviewable judgement rather than a fuzzy match.
EPICLIM_STRING_OVERLAY: dict[str, tuple[str, str]] = {
    "baleswar": ("OD-DIST-balasore", "alternate romanisation of Baleshwar/Balasore"),
    "nawapara": ("OD-DIST-nuapada", "alternate romanisation of Nuapada"),
    "berhampur": (
        "OD-DIST-ganjam",
        "city string; Berhampur (Brahmapur) lies in Ganjam district, so the report "
        "is attributed to its containing district",
    ),
}

DISEASE_GROUPS: dict[str, tuple[str, ...]] = {
    "any_reported_outbreak": (
        "Acute Diarrhoeal Disease",
        "Acute Gastroenteritis",
        "Cholera",
        "Chikungunya",
        "Dengue",
        "Malaria",
    ),
    "diarrhoeal_and_cholera": (
        "Acute Diarrhoeal Disease",
        "Acute Gastroenteritis",
        "Cholera",
    ),
    "vector_borne": ("Chikungunya", "Dengue", "Malaria"),
}


class TargetDataError(RuntimeError):
    """The catalogue is missing, altered, or contains an unresolvable district."""


@dataclass(frozen=True, slots=True)
class ReportedEvent:
    district_id: str
    district_string: str
    disease: str
    event_date: date
    week_start: date
    labelled_week: str


@dataclass(frozen=True, slots=True)
class TargetPanel:
    """Positive-only EpiClim row occurrence, indexed by event-date week."""

    group: str
    diseases: tuple[str, ...]
    events: tuple[ReportedEvent, ...]
    district_weeks: frozenset[tuple[str, date]]
    dataset_sha256: str
    resolution: dict[str, int]

    @property
    def positive_count(self) -> int:
        return len(self.district_weeks)

    def observed(self, district_id: str, week_start: date) -> int:
        return int((district_id, week_start) in self.district_weeks)


def _normalise(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def load_alias_index(path: Path | None = None) -> dict[str, str]:
    """Map normalised district strings to canonical district ids."""

    source = path or GAZETTEER_PATH
    index: dict[str, str] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            district_id = row["district_id"].strip()
            index[_normalise(row["canonical_name"])] = district_id
            for alias in row["aliases"].split("|"):
                if alias.strip():
                    index[_normalise(alias)] = district_id
    for raw, (district_id, _reason) in EPICLIM_STRING_OVERLAY.items():
        index[_normalise(raw)] = district_id
    return index


def week_start(value: date) -> date:
    """Monday of the ISO week containing ``value``."""

    calendar = value.isocalendar()
    return date.fromisocalendar(calendar.year, calendar.week, 1)


def load_reported_events(
    *,
    dataset: Path | None = None,
    alias_index: dict[str, str] | None = None,
    verify_digest: bool = True,
) -> tuple[tuple[ReportedEvent, ...], str, dict[str, int]]:
    path = dataset or DATASET_PATH
    if not path.exists():
        raise TargetDataError(
            f"EpiClim catalogue not found at {path}. Run "
            "`python scripts/audit_epiclim.py --save-dataset data/epiclim/Final_data.csv`."
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_digest and digest != EXPECTED_SHA256:
        raise TargetDataError(
            f"EpiClim catalogue digest {digest} does not match the audited "
            f"{EXPECTED_SHA256}; refusing to model an unverified vintage."
        )
    index = alias_index if alias_index is not None else load_alias_index()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    odisha = [row for row in rows if row["state_ut"].strip().casefold() == "odisha"]
    events: list[ReportedEvent] = []
    unresolved: Counter[str] = Counter()
    bad_dates = 0
    for row in odisha:
        district_string = row["district"].strip()
        district_id = index.get(_normalise(district_string))
        if district_id is None:
            unresolved[district_string] += 1
            continue
        try:
            event_date = date(int(row["year"]), int(row["mon"]), int(row["day"]))
        except ValueError:
            bad_dates += 1
            continue
        events.append(
            ReportedEvent(
                district_id=district_id,
                district_string=district_string,
                disease=row["Disease"].strip(),
                event_date=event_date,
                week_start=week_start(event_date),
                labelled_week=row["week_of_outbreak"].strip(),
            )
        )
    if unresolved:
        raise TargetDataError(
            "unresolved Odisha district strings (fail closed rather than drop "
            f"reports): {dict(unresolved)}"
        )
    resolution = {
        "odisha_rows": len(odisha),
        "resolved_rows": len(events),
        "unparseable_date_rows": bad_dates,
        "distinct_district_strings": len({row["district"].strip() for row in odisha}),
        "distinct_district_ids": len({event.district_id for event in events}),
        "overlay_rules_applied": len(EPICLIM_STRING_OVERLAY),
    }
    ordered = tuple(sorted(events, key=lambda item: (item.week_start, item.district_id)))
    return ordered, digest, resolution


def build_target_panel(
    group: str = "any_reported_outbreak",
    *,
    dataset: Path | None = None,
    alias_index: dict[str, str] | None = None,
    verify_digest: bool = True,
) -> TargetPanel:
    if group not in DISEASE_GROUPS:
        raise ValueError(f"unknown disease group {group!r}")
    diseases = DISEASE_GROUPS[group]
    events, digest, resolution = load_reported_events(
        dataset=dataset, alias_index=alias_index, verify_digest=verify_digest
    )
    selected = tuple(event for event in events if event.disease in diseases)
    district_weeks = frozenset((event.district_id, event.week_start) for event in selected)
    return TargetPanel(
        group=group,
        diseases=diseases,
        events=selected,
        district_weeks=district_weeks,
        dataset_sha256=digest,
        resolution={
            **resolution,
            "group_rows": len(selected),
            "group_distinct_district_weeks": len(district_weeks),
        },
    )
