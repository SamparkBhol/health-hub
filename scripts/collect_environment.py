#!/usr/bin/env python3
"""Capture NASA POWER environmental data in one of two modes.

``--mode receipt`` (default) captures and validates a single issue-time
environmental receipt. CHIRPS and ERA5 states are emitted alongside it. CHIRPS
is deliberately not fetched while the observed provider robots policy blocks
direct automation; ERA5 submission is not represented as retrieval.

``--mode historical-panel`` builds the modelling cache: one long validated daily
vintage per Odisha district, stored verbatim (gzipped) with a SHA-256 manifest
under ``data/environment/power_daily``. Pass explicit dates so the cache is
reproducible, for example::

    python scripts/collect_environment.py --mode historical-panel \\
        --start 2008-01-01 --end 2022-12-31

A reanalysis point sample is coarse environmental context for a district. It is
neither a district-average exposure nor disease surveillance.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.environmental.districts import load_district_points
from pipelines.environmental.historical import build_cache
from pipelines.environmental.models import AcquisitionState
from pipelines.environmental.nasa_power import fetch_power_daily, parse_power_daily
from pipelines.environmental.states import chirps_policy_state, era5_request_state
from workers.ingestion.safe_fetch import FetchError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "environment"
    / "nasa_power_bhubaneswar_demo_20260701_20260707.json"
)


def serialise(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: serialise(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, (tuple, list)):
        return [serialise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialise(item) for key, item in value.items()}
    return value


def fixture_receipt() -> Any:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return parse_power_daily(
        FIXTURE.read_bytes(),
        requested_url=payload["requested_url"],
        final_url=payload["requested_url"],
        retrieved_at=datetime.now(UTC),
        expected_longitude=85.82,
        expected_latitude=20.30,
        expected_start=date(2026, 7, 1),
        expected_end=date(2026, 7, 7),
        state=AcquisitionState.FIXTURE_FALLBACK,
    )


def environment_object_key(power: Any, *, captured_live: bool) -> str:
    retrieved_at = power.retrieved_at.astimezone(UTC)
    prefix = "environment/vintages" if captured_live else "environment/fixture-fallback"
    stamp = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}/{retrieved_at:%Y/%m/%d}/{stamp}_{power.sha256}.json"


def maybe_upload(path: Path, key: str, *, overwrite: bool = False) -> str | None:
    endpoint = os.getenv("R2_ENDPOINT_URL")
    bucket = os.getenv("R2_BUCKET")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not all((endpoint, bucket, access_key, secret_key)):
        return None
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    body = path.read_bytes()
    put_args: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": "application/json",
        "Metadata": {"content-sha256": hashlib.sha256(body).hexdigest()},
    }
    if not overwrite:
        put_args["IfNoneMatch"] = "*"
    client.put_object(
        **put_args,
    )
    return f"r2://{bucket}/{key}"


def historical_panel(args: argparse.Namespace) -> None:
    """Cache one long daily NASA POWER vintage per Odisha district.

    The cache is model input only.  A reanalysis point sample is environmental
    context, never disease surveillance and never a district-average exposure.
    """

    points = load_district_points()
    if args.district:
        wanted = {value.casefold() for value in args.district}
        points = tuple(
            point
            for point in points
            if point.district_id.casefold() in wanted or point.canonical_name.casefold() in wanted
        )
        if not points:
            raise SystemExit("no district matched --district")
    summary = build_cache(
        start=args.start,
        end=args.end,
        points=points,
        refresh=args.refresh,
        on_progress=(
            (lambda district_id, state: print(f"{state}\t{district_id}", file=sys.stderr))
            if args.verbose
            else None
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failures"]:
        raise SystemExit(1)


def current_conditions(args: argparse.Namespace) -> None:
    """Refresh the present-day, all-30-district environmental conditions layer.

    Three real retrievals, in order: near-real-time NASA POWER for the same 30
    representative points the model was trained on; every anonymously reachable
    IMD product; and the fitted environmental-suitability scoring of the two
    together.  Nothing is simulated and no case number is produced anywhere in
    this path.
    """

    from packages.forecasting.current_conditions import build_layer, write_layer
    from packages.forecasting.target import load_alias_index
    from pipelines.environmental.current import refresh_recent_cache
    from pipelines.environmental.imd import collect_live_imd

    progress = (
        (lambda district_id, state: print(f"{state}\t{district_id}", file=sys.stderr))
        if args.verbose
        else None
    )
    climate_summary: dict[str, Any] = {"skipped": True}
    if not args.skip_power:
        climate_summary = refresh_recent_cache(on_progress=progress)
        if args.verbose:
            failures = climate_summary.get("failures")
            failure_count = len(failures) if isinstance(failures, list) else 0
            print(
                f"power: {climate_summary['districts_cached']} districts, {failure_count} failures",
                file=sys.stderr,
            )

    imd_payload: dict[str, Any] | None = None
    if not args.skip_imd:
        names = {point.district_id: point.canonical_name for point in load_district_points()}
        imd_payload = collect_live_imd(
            alias_index=load_alias_index(), names=names, include_city=not args.skip_imd_city
        )
        if args.verbose:
            for product, rows in sorted(imd_payload["products"].items()):
                print(f"imd: {product} -> {len(rows)} rows", file=sys.stderr)

    layer = build_layer(imd_payload=imd_payload)
    path = write_layer(layer, args.output_layer)
    latest_key = os.getenv(
        "CURRENT_CONDITIONS_R2_KEY", "environment/current-conditions/latest.json"
    )
    object_uri = maybe_upload(path, latest_key, overwrite=True)
    print(
        json.dumps(
            {
                "output": str(path),
                "object_uri": object_uri,
                "as_of": layer["as_of"],
                "coverage": layer["coverage"],
                "data_edge": layer["data_edge"],
                "climate_failures": climate_summary.get("failures", []),
                "imd_failures": (imd_payload or {}).get("failures", []),
                "is_synthetic": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("receipt", "historical-panel", "current-conditions"),
        default="receipt",
        help=(
            "receipt: one issue-time validated vintage; historical-panel: "
            "30-district modelling cache; current-conditions: present-day "
            "30-district environmental conditions layer (NASA POWER + IMD)"
        ),
    )
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--skip-imd", action="store_true")
    parser.add_argument("--skip-imd-city", action="store_true")
    parser.add_argument(
        "--output-layer",
        type=Path,
        default=Path("data/environment/current_conditions.json"),
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2008, 1, 1))
    parser.add_argument("--district", action="append", default=[])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--longitude", type=float, default=85.82)
    parser.add_argument("--latitude", type=float, default=20.30)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=2))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--allow-fixture-fallback", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/environment/latest.json"))
    args = parser.parse_args()
    if args.mode == "historical-panel":
        historical_panel(args)
        return
    if args.mode == "current-conditions":
        current_conditions(args)
        return
    if not 1 <= args.days <= 31:
        raise SystemExit("--days must be between 1 and 31")
    start = args.end - timedelta(days=args.days - 1)
    provider_error = None
    captured_live = True
    try:
        power = fetch_power_daily(
            longitude=args.longitude,
            latitude=args.latitude,
            start=start,
            end=args.end,
        )
    except FetchError as exc:
        if not args.allow_fixture_fallback:
            raise
        captured_live = False
        provider_error = {"code": exc.code, "detail": str(exc)}
        power = fixture_receipt()

    now = datetime.now(UTC)
    report = {
        "schema_version": "1.0.0",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "captured_live": captured_live,
        "provider_error": provider_error,
        "nasa_power": serialise(power),
        "chirps": [
            serialise(chirps_policy_state(version="2.0", observed_at=now)),
            serialise(chirps_policy_state(version="3.0", observed_at=now)),
        ],
        "era5_land_t": serialise(
            era5_request_state(
                has_cds_credentials=bool(os.getenv("CDSAPI_KEY")),
                licence_accepted=os.getenv("CDS_LICENCE_ACCEPTED", "false").casefold() == "true",
                request_id=os.getenv("CDS_REQUEST_ID"),
                observed_at=now,
            )
        ),
        "warning": (
            "Fixture fallback proves parser continuity only and is not a newly captured vintage."
            if not captured_live
            else "NASA POWER is coarse environmental context, not disease surveillance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    object_uri = maybe_upload(
        args.output,
        environment_object_key(power, captured_live=captured_live),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "captured_live": captured_live,
                "object_uri": object_uri,
            }
        )
    )


if __name__ == "__main__":
    main()
