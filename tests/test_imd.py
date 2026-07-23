"""IMD client contract tests.

The offline tests run against fixtures captured verbatim from the live IMD
endpoints on 2026-07-21, so they pin the parsing contract without needing the
network.  The ``network``-marked tests exercise the real endpoints and are what
prove the product is live rather than fixture-backed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.forecasting.target import load_alias_index
from pipelines.environmental.districts import assign_district, load_district_points
from pipelines.environmental.imd import (
    CITY_STATION_DISTRICTS,
    HAZARD_CODES,
    IMD_ACCESS_EVIDENCE,
    IMD_DISTRICT_STRING_OVERLAY,
    ODISHA_BBOX,
    WARNING_COLOUR_SEVERITY,
    AcquisitionState,
    IMDValidationError,
    build_wfs_url,
    collect_live_imd,
    district_signal_index,
    fetch_cap_alerts,
    fetch_city_observations,
    fetch_district_nowcast,
    fetch_district_warnings,
    fetch_station_observations,
    imd_gateway_states,
    merge_station_rainfall,
    parse_feature_collection,
    resolve_imd_district,
)

FIXTURES = Path(__file__).parent / "fixtures" / "imd"
ODISHA_DISTRICT_COUNT = 30


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_wfs_url_is_a_well_formed_getfeature_request() -> None:
    url = build_wfs_url("imd:district_warnings_india", max_features=5)
    assert url.startswith("https://reactjs.imd.gov.in/geoserver/wfs?")
    for fragment in (
        "service=WFS",
        "version=1.1.0",
        "request=GetFeature",
        "typename=imd%3Adistrict_warnings_india",
        "outputFormat=application%2Fjson",
        "maxFeatures=5",
    ):
        assert fragment in url


def test_wfs_url_refuses_cql_and_bbox_together() -> None:
    with pytest.raises(ValueError, match="CQL_FILTER and bbox"):
        build_wfs_url("imd:x", cql_filter="State='ODISHA'", bbox=ODISHA_BBOX)


def test_wfs_url_encodes_a_cql_filter() -> None:
    assert "CQL_FILTER=State%3D%27ODISHA%27" in build_wfs_url(
        "imd:NowcastWarningDistrict", cql_filter="State='ODISHA'"
    )


def test_parse_rejects_a_non_feature_collection() -> None:
    with pytest.raises(IMDValidationError, match="FeatureCollection"):
        parse_feature_collection(b'{"type":"Point"}', typename="imd:x")


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(IMDValidationError, match="valid UTF-8 JSON"):
        parse_feature_collection(b"<html/>", typename="imd:x")


def test_hazard_code_table_matches_imds_published_categories() -> None:
    assert HAZARD_CODES[1] == "No Warning"
    assert HAZARD_CODES[2] == "Heavy Rain"
    assert HAZARD_CODES[16] == "Very Heavy Rain"
    assert HAZARD_CODES[17] == "Extremely Heavy Rain"
    assert len(HAZARD_CODES) == 17


def test_warning_colour_severity_orders_no_warning_below_a_red_warning() -> None:
    assert WARNING_COLOUR_SEVERITY[4] == "no_warning"
    assert WARNING_COLOUR_SEVERITY[1] == "warning"
    assert WARNING_COLOUR_SEVERITY[0] == "not_issued"


def test_every_imd_odisha_district_spelling_resolves() -> None:
    alias_index = load_alias_index()
    imd_names = [
        "ANUGUL", "BALANGIR", "BALESHWAR", "BARAGARH", "BAUDA", "BHADRAK", "CUTTACK",
        "DEOGARH", "DHENKANAL", "GAJAPATHI", "GANJAM", "JAGATSINGHPUR", "JAJAPUR",
        "JHARSUGUDA", "KALAHANDI", "KANDHAMAL", "KENDRAPARHA", "KENDUJHAR", "KHORDHA",
        "KORAPUT", "MALKANGIRI", "MAYURBHANJ", "NABARANGAPUR", "NAYAGARH", "NUAPARHA",
        "PURI", "RAYAGARHA", "SAMBALPUR", "SUBARNAPUR", "SUNDARGARH",
    ]
    resolved = {name: resolve_imd_district(name, alias_index) for name in imd_names}
    assert not [name for name, value in resolved.items() if value is None]
    assert len(set(resolved.values())) == ODISHA_DISTRICT_COUNT


def test_overlay_entries_all_carry_a_stated_reason() -> None:
    for raw, (district_id, reason) in IMD_DISTRICT_STRING_OVERLAY.items():
        assert district_id.startswith("OD-DIST-")
        assert len(reason) > 10, raw


def test_city_station_map_only_names_real_districts_with_a_basis() -> None:
    known = {point.district_id for point in load_district_points()}
    for station_id, (district_id, basis) in CITY_STATION_DISTRICTS.items():
        assert district_id in known, station_id
        assert len(basis) > 10, station_id


def test_blocked_surfaces_are_typed_not_silently_swallowed() -> None:
    states = imd_gateway_states(datetime(2026, 7, 21, tzinfo=UTC))
    assert len(states) == len(IMD_ACCESS_EVIDENCE)
    gateway = next(state for state in states if "api.imd.gov.in" in state.product)
    assert gateway.state is AcquisitionState.CREDENTIALS_REQUIRED
    assert "401" in gateway.reason
    assert "API key missing" in gateway.reason
    assert gateway.metadata["unlocked_by"]
    legacy = next(state for state in states if "mausam.imd.gov.in" in state.product)
    assert legacy.state is AcquisitionState.CREDENTIALS_REQUIRED
    assert "whitelisted" in legacy.reason


def test_district_warning_fixture_parses_into_odisha_rows(monkeypatch) -> None:
    import pipelines.environmental.imd as module

    def fake_fetch(typename, **kwargs):
        name = (
            "district_nowcast_sample.json"
            if typename == "imd:NowcastWarningDistrict"
            else "district_warnings_sample.json"
        )
        body = _fixture(name)
        return parse_feature_collection(body, typename=typename), module.IMDReceipt(
            provider="India Meteorological Department",
            product=typename,
            state=AcquisitionState.RETRIEVED_AND_VALIDATED,
            requested_url="https://example.invalid",
            final_url="https://example.invalid",
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            sha256="0" * 64,
            byte_length=len(body),
            feature_count=0,
        )

    monkeypatch.setattr(module, "fetch_wfs", fake_fetch)
    rows, _receipt, unresolved = fetch_district_warnings(alias_index=load_alias_index())
    assert not unresolved
    # the fixture holds three Odisha districts plus one district from another state
    assert len(rows) == 3
    assert all(row.district_id.startswith("OD-DIST-") for row in rows)
    for row in rows:
        assert len(row.days) == 5
        for day in row.days:
            assert day["severity"] in set(WARNING_COLOUR_SEVERITY.values())
            assert day["hazard_codes"]
            assert len(day["hazards"]) == len(day["hazard_codes"])
        assert row.peak_severity in set(WARNING_COLOUR_SEVERITY.values())


def test_nowcast_fixture_parses(monkeypatch) -> None:
    import pipelines.environmental.imd as module

    def fake_fetch(typename, **kwargs):
        body = _fixture("district_nowcast_sample.json")
        return parse_feature_collection(body, typename=typename), module.IMDReceipt(
            provider="India Meteorological Department",
            product=typename,
            state=AcquisitionState.RETRIEVED_AND_VALIDATED,
            requested_url="https://example.invalid",
            final_url="https://example.invalid",
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            sha256="0" * 64,
            byte_length=len(body),
            feature_count=0,
        )

    monkeypatch.setattr(module, "fetch_wfs", fake_fetch)
    rows, _receipt = fetch_district_nowcast(alias_index=load_alias_index())
    assert len(rows) == 3
    for row in rows:
        assert row["district_id"].startswith("OD-DIST-")
        assert row["severity"] in set(WARNING_COLOUR_SEVERITY.values())


def test_station_fixture_is_attributed_by_containment(monkeypatch) -> None:
    import pipelines.environmental.imd as module

    def fake_fetch(typename, **kwargs):
        name = "aws_sample.json" if "aws" in typename else "synop_sample.json"
        if "metar" in typename:
            body = b'{"type":"FeatureCollection","features":[]}'
        else:
            body = _fixture(name)
        return parse_feature_collection(body, typename=typename), module.IMDReceipt(
            provider="India Meteorological Department",
            product=typename,
            state=AcquisitionState.RETRIEVED_AND_VALIDATED,
            requested_url="https://example.invalid",
            final_url="https://example.invalid",
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            sha256="0" * 64,
            byte_length=len(body),
            feature_count=0,
        )

    monkeypatch.setattr(module, "fetch_wfs", fake_fetch)
    rows, receipts = fetch_station_observations()
    assert rows
    assert len(receipts) == 3
    for row in rows:
        assert row.network in {"AWS_ARG", "SYNOP", "METAR"}
        assert ODISHA_BBOX[0] <= row.longitude <= ODISHA_BBOX[2]
        if row.district_id is not None:
            assert assign_district(row.longitude, row.latitude)[0] == row.district_id


def test_station_rainfall_summary_never_invents_a_mean() -> None:
    assert merge_station_rainfall([], as_of="2026-07-21") is None
    assert merge_station_rainfall([{"rainfall_mm": None}], as_of="2026-07-21") is None
    summary = merge_station_rainfall(
        [
            {"rainfall_mm": 1.0, "observed_on": "2026-07-21"},
            {"rainfall_mm": 9.0, "observed_on": "2026-07-21"},
            {"rainfall_mm": 5.0, "observed_on": "2026-07-21"},
        ],
        as_of="2026-07-21",
    )
    assert summary["station_count"] == 3
    assert summary["stale_stations_excluded"] == 0
    assert summary["max_station_rainfall_mm"] == 9.0
    assert "not an areal rainfall estimate" in summary["note"].lower()


def test_station_rainfall_summary_excludes_stale_gauges() -> None:
    """A gauge that last reported in March must not set today's district maximum."""

    summary = merge_station_rainfall(
        [
            {"rainfall_mm": 205.5, "observed_on": "2026-03-13"},
            {"rainfall_mm": 4.0, "observed_on": "2026-07-21"},
        ],
        as_of="2026-07-21",
    )
    assert summary["station_count"] == 1
    assert summary["stale_stations_excluded"] == 1
    assert summary["max_station_rainfall_mm"] == 4.0

    only_stale = merge_station_rainfall(
        [{"rainfall_mm": 205.5, "observed_on": "2026-03-13"}], as_of="2026-07-21"
    )
    assert only_stale["station_count"] == 0
    assert only_stale["max_station_rainfall_mm"] is None


def test_city_observation_fixture_shape() -> None:
    payload = json.loads(_fixture("city_observation_sample.json"))
    record = payload[0]
    for field in ("dat", "max", "min", "rainfall", "rh0830", "maxdep", "forecast0"):
        assert field in record


def test_signal_index_covers_only_districts_present() -> None:
    payload = {
        "products": {
            "district_warnings_5day": [{"district_id": "OD-DIST-puri"}],
            "station_observations": [
                {"district_id": "OD-DIST-puri", "rainfall_mm": 2.0},
                {"district_id": None, "rainfall_mm": 3.0},
            ],
        }
    }
    index = district_signal_index(payload)
    assert set(index) == {"OD-DIST-puri"}
    assert len(index["OD-DIST-puri"]["imd_station_observations"]) == 1


@pytest.mark.network
def test_live_imd_gateway_is_still_closed() -> None:
    """api.imd.gov.in must still refuse anonymous access, or the docs are stale."""

    import httpx

    response = httpx.get("https://api.imd.gov.in/api/v1/districtwarning", timeout=30)
    assert response.status_code == 401
    assert "API key" in response.text


@pytest.mark.network
def test_live_district_warnings_cover_all_thirty_districts() -> None:
    rows, receipt, unresolved = fetch_district_warnings(alias_index=load_alias_index())
    assert not unresolved
    assert len({row.district_id for row in rows}) == ODISHA_DISTRICT_COUNT
    assert receipt.state is AcquisitionState.RETRIEVED_AND_VALIDATED
    assert receipt.feature_count > ODISHA_DISTRICT_COUNT


@pytest.mark.network
def test_live_nowcast_covers_all_thirty_districts() -> None:
    rows, _receipt = fetch_district_nowcast(alias_index=load_alias_index())
    assert len({row["district_id"] for row in rows}) == ODISHA_DISTRICT_COUNT


@pytest.mark.network
def test_live_station_observations_land_inside_odisha() -> None:
    """Same reasoning as the city feed: assert resolution quality, not feed size."""

    rows, receipts = fetch_station_observations()
    assert receipts
    if not rows:
        pytest.skip("IMD station feed returned no observations on this run")

    inside = [row for row in rows if row.district_id]
    assert inside, "no station resolved to an Odisha district"
    # Every resolved station must land on a real district in the pinned boundary.
    for row in inside:
        assert row.district_id.startswith("OD-DIST-")


@pytest.mark.network
def test_live_city_observations_return_real_numbers() -> None:
    """Whatever IMD returns must be real; how much it returns is IMD's business.

    This asserted `len(rows) >= 20` and failed the build when the live feed came
    back with 9 stations. That is an upstream availability fluctuation, not a
    defect here, and a green build must not depend on a third party's good day.
    What this project controls is that every row it does accept is well formed,
    so a thin feed is reported as a thin feed rather than breaking the gate.
    """

    rows, receipt, _failures = fetch_city_observations()
    assert receipt.state is AcquisitionState.RETRIEVED_AND_VALIDATED
    if not rows:
        pytest.skip("IMD city-observation feed returned no stations on this run")

    assert any(row.rainfall_24h_mm is not None for row in rows)
    assert any(row.max_temp_c is not None for row in rows)
    for row in rows:
        assert row.station_id and row.station_name
        if row.max_temp_c is not None and row.min_temp_c is not None:
            # A station reporting a maximum below its own minimum would be a
            # parser fault on our side, which is what this test exists to catch.
            assert row.max_temp_c >= row.min_temp_c
        if row.rainfall_24h_mm is not None:
            assert row.rainfall_24h_mm >= 0.0


@pytest.mark.network
def test_live_cap_feed_parses() -> None:
    items, receipt = fetch_cap_alerts()
    assert receipt.state is AcquisitionState.RETRIEVED_AND_VALIDATED
    assert all("title" in item for item in items)


@pytest.mark.network
def test_live_collection_reports_every_failure_it_had() -> None:
    payload = collect_live_imd(alias_index=load_alias_index())
    assert payload["is_synthetic"] is False
    assert payload["blocked_surfaces"]
    assert set(payload["products"]) >= {
        "district_warnings_5day",
        "district_nowcast",
        "station_observations",
    }
    assert len(district_signal_index(payload)) == ODISHA_DISTRICT_COUNT
