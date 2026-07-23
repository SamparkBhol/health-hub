from __future__ import annotations

import csv
from datetime import date

from fastapi.testclient import TestClient

from packages.forecasting.authorised_surveillance import REQUIRED_COLUMNS
from packages.forecasting.operational import observed_surveillance_map, train
from services.api.main import create_app


def _write_two_vintages(path) -> None:  # noqa: ANN001 - pytest path fixture
    common = {
        "district_id": "OD-DIST-khordha",
        "disease": "dengue",
        "week_start": "2026-07-06",
        "population": "100000",
        "reporting_units_expected": "10",
        "reporting_units_received": "9",
        "case_volume_completeness": "0.9",
        "case_definition_version": "ihip-dengue-v1",
        "outbreak_threshold_per_100k": "2.0",
        "threshold_version": "dengue-threshold-v1",
        "source_vintage": "authorised-export-v1",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({**common, "cases": "1", "known_at": "2026-07-13"})
        writer.writerow({**common, "cases": "4", "known_at": "2026-07-20"})


def test_observed_map_is_bitemporal_and_uses_rate_not_article_density(tmp_path) -> None:
    source = tmp_path / "district_week.csv"
    _write_two_vintages(source)

    earlier = observed_surveillance_map(path=source, as_of=date(2026, 7, 15))
    latest = observed_surveillance_map(path=source)

    assert earlier["metric"] == "rate_per_100k"
    assert earlier["records"][0]["cases"] == 1
    assert earlier["records"][0]["map_value"] == 1.0
    assert latest["records"][0]["cases"] == 4
    assert latest["records"][0]["map_value"] == 4.0
    assert latest["records"][0]["observation_state"] == "observed_complete"


def test_operational_train_refuses_missing_authorised_export(tmp_path) -> None:
    payload = train(path=tmp_path / "missing.csv")
    assert payload["status"] == "insufficient_evidence"
    assert payload["results"] == []
    assert payload["reason_codes"] == ["AUTHORISED_DISTRICT_WEEK_EXPORT_NOT_PRESENT"]


def test_operational_routes_fail_closed_without_external_state(tmp_path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path}/api.db"))

    observed = client.get("/api/v1/observed-surveillance/map").json()
    assert observed["data"]["records"] == []
    assert observed["context"]["coverage_state"] == "awaiting_sponsor_data"

    summary = client.get("/api/v1/forecast/operational").json()
    assert summary["data"]["cells"] == []

    forecast = client.get(
        "/api/v1/forecast/operational/map",
        params={"disease": "dengue", "horizon_weeks": 1},
    ).json()
    assert forecast["data"]["status"] == "insufficient_evidence"
    assert forecast["data"]["districts"] == []
