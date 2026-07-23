from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from pipelines.environmental.models import AcquisitionState
from pipelines.environmental.nasa_power import (
    EnvironmentalValidationError,
    build_power_url,
    parse_power_daily,
)
from pipelines.environmental.states import chirps_policy_state, era5_request_state
from scripts.collect_environment import environment_object_key, fixture_receipt

FIXTURE = (
    Path(__file__).parent
    / "fixtures/environment/nasa_power_bhubaneswar_demo_20260701_20260707.json"
)


def test_environment_archive_key_is_append_only_and_segregates_fixtures() -> None:
    receipt = fixture_receipt()
    live_key = environment_object_key(receipt, captured_live=True)
    fixture_key = environment_object_key(receipt, captured_live=False)
    assert live_key.startswith("environment/vintages/")
    assert fixture_key.startswith("environment/fixture-fallback/")
    assert receipt.sha256 in live_key
    assert receipt.retrieved_at.strftime("%Y%m%dT%H%M%S%fZ") in live_key
    assert live_key != fixture_key


def test_nasa_power_fixture_creates_typed_receipt() -> None:
    body = FIXTURE.read_bytes()
    payload = json.loads(body)
    receipt = parse_power_daily(
        body,
        requested_url=payload["requested_url"],
        final_url=payload["requested_url"],
        retrieved_at=datetime(2026, 7, 21, 12, tzinfo=UTC),
        expected_longitude=85.82,
        expected_latitude=20.30,
        expected_start=date(2026, 7, 1),
        expected_end=date(2026, 7, 7),
        state=AcquisitionState.FIXTURE_FALLBACK,
    )
    assert receipt.state == AcquisitionState.FIXTURE_FALLBACK
    assert len(receipt.values) == 35
    assert receipt.time_standard == "UTC"
    assert receipt.sha256
    assert "not_an_authoritative_district_centroid" in receipt.warnings[1]
    rainfall = [item for item in receipt.values if item.parameter == "PRECTOTCORR"]
    assert rainfall[0].unit == "mm/day"
    assert rainfall[0].value == 12.61


def test_nasa_power_contract_rejects_unit_drift() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["parameters"]["T2M"]["units"] = "kelvin"
    body = json.dumps(payload).encode()
    with pytest.raises(EnvironmentalValidationError, match="unit mismatch"):
        parse_power_daily(
            body,
            requested_url=payload["requested_url"],
            final_url=payload["requested_url"],
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            expected_longitude=85.82,
            expected_latitude=20.30,
            expected_start=date(2026, 7, 1),
            expected_end=date(2026, 7, 7),
        )


def test_power_url_is_deterministic_and_utc() -> None:
    url = build_power_url(
        longitude=85.82,
        latitude=20.30,
        start=date(2026, 7, 1),
        end=date(2026, 7, 7),
    )
    assert url.startswith("https://power.larc.nasa.gov/api/temporal/daily/point?")
    assert "time-standard=UTC" in url
    assert "PRECTOTCORR" in url


def test_chirps_fails_closed_on_observed_policy() -> None:
    state = chirps_policy_state(version="3.0", observed_at=datetime(2026, 7, 21, tzinfo=UTC))
    assert state.state == AcquisitionState.AWAITING_SOURCE_PERMISSION_OR_APPROVED_API
    assert not state.metadata["direct_data_host_automation"]


@pytest.mark.parametrize(
    ("credentials", "accepted", "request_id", "expected"),
    [
        (False, False, None, AcquisitionState.CREDENTIALS_REQUIRED),
        (True, False, None, AcquisitionState.LICENCE_ACCEPTANCE_REQUIRED),
        (True, True, None, AcquisitionState.NOT_REQUESTED),
        (True, True, "cds-request-123", AcquisitionState.REQUEST_SUBMITTED),
    ],
)
def test_era_states_are_not_confused_with_retrieval(
    credentials, accepted, request_id, expected
) -> None:
    state = era5_request_state(
        has_cds_credentials=credentials,
        licence_accepted=accepted,
        request_id=request_id,
        observed_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    assert state.state == expected
    assert state.archive_started_no_history_before is None
