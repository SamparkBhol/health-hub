"""Tests for district representative points and the historical climate cache."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from packages.forecasting.climate import load_weekly_panel, weekly_from_receipt
from pipelines.environmental.districts import (
    POINT_WARNINGS,
    _point_in_ring,
    _representative_point,
    load_district_points,
)
from pipelines.environmental.historical import (
    CACHE_ROOT,
    CachedVintage,
    ClimateCacheError,
    load_receipt,
    object_name,
    read_manifest,
    store_body,
    write_manifest,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures/environment/nasa_power_bhubaneswar_demo_20260701_20260707.json"
)
requires_cache = pytest.mark.skipif(
    not (CACHE_ROOT / "manifest.json").exists(),
    reason="historical NASA POWER cache not built",
)


def test_every_odisha_district_gets_an_interior_point() -> None:
    points = load_district_points()
    assert len(points) == 30
    assert len({point.district_id for point in points}) == 30
    for point in points:
        # Odisha's real extent, so a point outside it is a boundary or maths bug.
        assert 81.0 < point.longitude < 88.0
        assert 17.5 < point.latitude < 22.7


def test_representative_point_falls_inside_a_concave_ring() -> None:
    # A C-shaped ring whose area centroid lies in the notch, outside the polygon.
    ring = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 3.0),
        (3.0, 3.0),
        (3.0, 7.0),
        (10.0, 7.0),
        (10.0, 10.0),
        (0.0, 10.0),
    ]
    longitude, latitude, method = _representative_point([ring])
    assert _point_in_ring(longitude, latitude, ring)
    assert method == "max_clearance_interior_grid_probe"


def test_point_warnings_deny_the_district_average_reading() -> None:
    joined = " ".join(POINT_WARNINGS)
    assert "not_district_average_exposure" in joined
    assert "not_administrative_headquarter" in joined


def test_cache_round_trip_validates_the_provider_contract(tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    payload = json.loads(body)
    point = type(
        "P",
        (),
        {
            "district_id": "OD-DIST-test",
            "canonical_name": "Test",
            "longitude": 85.82,
            "latitude": 20.30,
            "method": "fixture",
        },
    )()
    vintage = store_body(
        body,
        point=point,
        start=date(2026, 7, 1),
        end=date(2026, 7, 7),
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        requested_url=payload["requested_url"],
        api_version="v2.9.4",
        root=tmp_path,
    )
    write_manifest([vintage], tmp_path)
    reloaded = read_manifest(tmp_path)
    assert reloaded["OD-DIST-test"] == vintage
    receipt = load_receipt(reloaded["OD-DIST-test"], tmp_path)
    assert len(receipt.values) == 35
    assert receipt.start == date(2026, 7, 1)


def test_tampered_cache_object_fails_closed(tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    name = object_name("OD-DIST-test", date(2026, 7, 1), date(2026, 7, 7))
    (tmp_path / name).write_bytes(gzip.compress(body + b" "))
    vintage = CachedVintage(
        district_id="OD-DIST-test",
        canonical_name="Test",
        longitude=85.82,
        latitude=20.30,
        point_method="fixture",
        start=date(2026, 7, 1),
        end=date(2026, 7, 7),
        sha256="0" * 64,
        byte_length=len(body),
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        api_version="v2.9.4",
        requested_url="https://power.larc.nasa.gov/api/temporal/daily/point",
        object_name=name,
    )
    with pytest.raises(ClimateCacheError, match="does not match manifest"):
        load_receipt(vintage, tmp_path)


def test_missing_cache_object_fails_closed(tmp_path: Path) -> None:
    vintage = CachedVintage(
        district_id="OD-DIST-ghost",
        canonical_name="Ghost",
        longitude=85.0,
        latitude=20.0,
        point_method="fixture",
        start=date(2026, 7, 1),
        end=date(2026, 7, 7),
        sha256="0" * 64,
        byte_length=1,
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        api_version="v2.9.4",
        requested_url="https://power.larc.nasa.gov/api/temporal/daily/point",
        object_name="absent.json.gz",
    )
    with pytest.raises(ClimateCacheError, match="missing"):
        load_receipt(vintage, tmp_path)


def test_weekly_aggregation_of_the_bundled_fixture() -> None:
    from pipelines.environmental.nasa_power import parse_power_daily

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    receipt = parse_power_daily(
        FIXTURE.read_bytes(),
        requested_url=payload["requested_url"],
        final_url=payload["requested_url"],
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        expected_longitude=85.82,
        expected_latitude=20.30,
        expected_start=date(2026, 7, 1),
        expected_end=date(2026, 7, 7),
    )
    weeks = weekly_from_receipt(receipt)
    # 1-7 July 2026 straddles two ISO weeks, so neither is complete.
    assert all(not item.complete for item in weeks)


def test_empty_cache_directory_raises_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="collect_environment"):
        load_weekly_panel(tmp_path)


@requires_cache
def test_shipped_cache_covers_every_district_over_the_modelling_window() -> None:
    manifest = read_manifest()
    assert len(manifest) == 30
    assert {vintage.district_id for vintage in manifest.values()} == {
        point.district_id for point in load_district_points()
    }
    for vintage in manifest.values():
        assert vintage.start <= date(2008, 1, 1)
        assert vintage.end >= date(2022, 12, 31)
        assert len(vintage.sha256) == 64
