from __future__ import annotations

import csv
from datetime import date, timedelta

import pytest

from packages.forecasting.authorised_surveillance import (
    REQUIRED_COLUMNS,
    SurveillanceContractError,
    audit_export,
    load_export,
)


def _write_export(path, *, complete: bool = True) -> None:
    districts = [
        "OD-DIST-angul",
        "OD-DIST-balasore",
        "OD-DIST-bargarh",
        "OD-DIST-bhadrak",
        "OD-DIST-balangir",
    ]
    start = date(2023, 1, 2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for district in districts:
            for offset in range(156):
                if not complete and district == districts[0] and offset == 20:
                    continue
                week = start + timedelta(weeks=offset)
                writer.writerow(
                    {
                        "district_id": district,
                        "disease": "dengue",
                        "week_start": week.isoformat(),
                        "cases": str(offset % 3),
                        "population": "100000",
                        "reporting_units_expected": "10",
                        "reporting_units_received": "9",
                        "case_volume_completeness": "0.9",
                        "known_at": (week + timedelta(days=9)).isoformat(),
                        "case_definition_version": "ihip-dengue-v1",
                        "outbreak_threshold_per_100k": "2.0",
                        "threshold_version": "dengue-endemic-channel-2026-v1",
                        "source_vintage": "odisha-ihip-weekly-2026-07-22",
                    }
                )


def test_absent_export_reports_a_precise_gate(tmp_path) -> None:
    payload = audit_export(tmp_path / "missing.csv")
    assert payload["eligible_for_training"] is False
    assert payload["status"] == "awaiting_authorised_aggregate_export"
    assert payload["reason_codes"] == ["AUTHORISED_DISTRICT_WEEK_EXPORT_NOT_PRESENT"]
    assert "cases" in payload["required_columns"]


def test_aggregate_contract_rejects_non_monday_week(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        ",".join(REQUIRED_COLUMNS)
        + (
            "\nOD-DIST-angul,dengue,2023-01-03,0,100000,1,1,1.0,2023-01-10,"
            "v1,2.0,threshold-v1,source-v1\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(SurveillanceContractError, match="ISO Monday"):
        load_export(path)


def test_complete_aggregate_panel_is_eligible_for_the_training_stage(tmp_path) -> None:
    path = tmp_path / "district_week.csv"
    _write_export(path)
    rows = load_export(path)
    assert len(rows) == 5 * 156
    assert rows[0].rate_per_100k >= 0.0
    payload = audit_export(path)
    assert payload["eligible_for_training"] is True
    assert payload["diseases"]["dengue"]["districts_meeting_history_and_completeness_floor"] == 5


def test_missing_week_is_not_silently_treated_as_zero(tmp_path) -> None:
    path = tmp_path / "gappy.csv"
    _write_export(path, complete=False)
    payload = audit_export(path)
    assert payload["eligible_for_training"] is False
    assert payload["diseases"]["dengue"]["districts_with_explicit_nil_or_case_row_every_week"] == 4
