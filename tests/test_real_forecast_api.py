"""API-surface tests for the experimental EpiClim row-occurrence model.

These guard the properties that make Objective 3 defensible rather than merely
present: the quantity is membership in a frozen incomplete file and never
incidence or official publication, every district in the retained historical
map joins the pinned boundary, and a failed cell emits no number.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from services.api.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _client(tmp_path) -> TestClient:  # noqa: ANN001 - pytest fixture value
    return TestClient(create_app(database_url=f"sqlite:///{tmp_path}/api.db"))


def test_real_forecast_summary_defers_every_refused_cell(tmp_path) -> None:
    payload = _client(tmp_path).get("/api/v1/forecast/real").json()

    assert payload["data"]["is_synthetic"] is False
    assert payload["data"]["is_incidence"] is False
    assert payload["data"]["is_case_count"] is False
    assert payload["data"]["experimental"] is True
    assert payload["data"]["is_official_publication_probability"] is False
    assert payload["data"]["published_cells"] == []
    assert "NOT_INCIDENCE" in {item["code"] for item in payload["warnings"]}

    refused = payload["data"]["refused_cells"]
    assert refused, "the artefact must retain at least one honestly refused cell"
    deferred = {item["capability"] for item in payload["deferrals"]}
    for cell in refused:
        capability = (
            f"epiclim_catalogue_row_experiment:{cell['disease_group']}:{cell['horizon_weeks']}w"
        )
        assert capability in deferred, f"{capability} refused but not deferred"


def test_experimental_map_joins_boundary_and_refuses_disease_risk_reading(tmp_path) -> None:
    response = _client(tmp_path).get(
        "/api/v1/forecast/real/map",
        params={"disease_group": "any_reported_outbreak", "horizon_weeks": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    data = payload["data"]
    assert data["status"] == "experimental"
    assert data["quantity"] == "experimental_epiclim_catalogue_row_occurrence"
    assert data["experimental"] is True
    assert data["is_operational_forecast"] is False

    boundary = json.loads(
        (PROJECT_ROOT / "data" / "boundaries" / "odisha_districts_census_2011.geojson").read_text(
            encoding="utf-8"
        )
    )
    boundary_ids = {feature["properties"]["district_id"] for feature in boundary["features"]}
    mapped_ids = {row["district_id"] for row in data["districts"]}
    assert mapped_ids == boundary_ids, "every district must join the pinned boundary"

    for row in data["districts"]:
        assert 0.0 <= row["probability_epiclim_catalogue_row"] <= 1.0
        assert 0.0 <= row["seasonal_baseline_probability"] <= 1.0

    codes = {item["code"] for item in payload["warnings"]}
    assert {"NOT_INCIDENCE", "HISTORICAL_REISSUE"} <= codes


def test_environment_detail_maps_external_state_without_500(tmp_path) -> None:
    response = _client(tmp_path).get("/api/v1/environment/current")
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["layer_type"] == "environment"
    assert payload["data"]["coverage"]["districts"] == 30
    assert all(
        item["state"]["code"]
        in {
            "awaiting_external_credential",
            "awaiting_source_permission_or_approved_api",
            "source_temporarily_unavailable",
        }
        for item in payload["deferrals"]
    )


def test_cell_without_skill_refuses_rather_than_emitting_a_number(tmp_path) -> None:
    response = _client(tmp_path).get(
        "/api/v1/forecast/real/map",
        params={"disease_group": "any_reported_outbreak", "horizon_weeks": 12},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["status"] == "insufficient_evidence"
    assert payload["data"]["districts"] == []
    assert payload["context"]["coverage_state"] == "unavailable"
    assert payload["deferrals"], "a refused cell must carry a typed deferral"


def test_unsupported_group_and_horizon_are_rejected(tmp_path) -> None:
    client = _client(tmp_path)
    assert (
        client.get(
            "/api/v1/forecast/real/map",
            params={"disease_group": "not_a_group", "horizon_weeks": 1},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/forecast/real/map",
            params={"disease_group": "vector_borne", "horizon_weeks": 3},
        ).status_code
        == 422
    )


def test_incidence_forecast_is_still_refused(tmp_path) -> None:
    """The real occurrence model must not quietly satisfy the incidence route."""

    payload = _client(tmp_path).get("/api/v1/forecast").json()
    assert payload["data"] == []
    assert "NOT_A_FORECAST" in {item["code"] for item in payload["warnings"]}


def test_signal_map_exposes_all_thirty_districts_without_inventing_zeroes(
    tmp_path,
) -> None:
    payload = _client(tmp_path).get("/api/v1/maps/published-signals").json()
    data = payload["data"]

    assert data["district_universe_size"] == 30
    assert len(data["district_universe"]) == 30

    for entry in data["district_universe"]:
        if entry["observation_state"] == "observed":
            assert entry["published_signal_count"] > 0
        else:
            # An unknown district must carry no count at all -- never a zero, which
            # would read as "no disease here".
            assert "published_signal_count" not in entry
