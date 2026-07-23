from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
from datetime import date

from workers.ingestion.safe_fetch import FetchPolicy, fetch_url

from .models import AcquisitionState, EnvironmentalReceipt, EnvironmentalValue

POWER_HOST = "power.larc.nasa.gov"
POWER_ENDPOINT = f"https://{POWER_HOST}/api/temporal/daily/point"
PARAMETERS = ("PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M")
EXPECTED_UNITS = {
    "PRECTOTCORR": "mm/day",
    "T2M": "C",
    "T2M_MAX": "C",
    "T2M_MIN": "C",
    "RH2M": "%",
}


class EnvironmentalValidationError(ValueError):
    pass


def build_power_url(*, longitude: float, latitude: float, start: date, end: date) -> str:
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("invalid point coordinates")
    if end < start:
        raise ValueError("end must not precede start")
    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": f"{longitude:.6f}",
            "latitude": f"{latitude:.6f}",
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }
    )
    return f"{POWER_ENDPOINT}?{query}"


def parse_power_daily(
    body: bytes,
    *,
    requested_url: str,
    final_url: str,
    retrieved_at,
    expected_longitude: float,
    expected_latitude: float,
    expected_start: date,
    expected_end: date,
    state: AcquisitionState = AcquisitionState.RETRIEVED_AND_VALIDATED,
) -> EnvironmentalReceipt:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentalValidationError("POWER response is not valid UTF-8 JSON") from exc
    try:
        coordinates = payload["geometry"]["coordinates"]
        header = payload["header"]
        parameter_values = payload["properties"]["parameter"]
        parameter_metadata = payload["parameters"]
    except (KeyError, TypeError) as exc:
        raise EnvironmentalValidationError("POWER response is missing required fields") from exc
    if not math.isclose(
        float(coordinates[0]), expected_longitude, abs_tol=0.01
    ) or not math.isclose(float(coordinates[1]), expected_latitude, abs_tol=0.01):
        raise EnvironmentalValidationError(
            "POWER response point does not match the requested point"
        )
    if header.get("time_standard") != "UTC":
        raise EnvironmentalValidationError("POWER response is not UTC")
    if header.get("start") != expected_start.strftime("%Y%m%d") or header.get(
        "end"
    ) != expected_end.strftime("%Y%m%d"):
        raise EnvironmentalValidationError("POWER response date range does not match the request")
    fill_value = float(header.get("fill_value", -999.0))
    values: list[EnvironmentalValue] = []
    for parameter in PARAMETERS:
        if parameter not in parameter_values or parameter not in parameter_metadata:
            raise EnvironmentalValidationError(f"POWER response omitted {parameter}")
        unit = str(parameter_metadata[parameter].get("units"))
        if unit != EXPECTED_UNITS[parameter]:
            raise EnvironmentalValidationError(
                f"POWER unit mismatch for {parameter}: "
                f"expected {EXPECTED_UNITS[parameter]!r}, received {unit!r}"
            )
        observed_days = parameter_values[parameter]
        for raw_day, raw_value in observed_days.items():
            try:
                day = date.fromisoformat(f"{raw_day[:4]}-{raw_day[4:6]}-{raw_day[6:8]}")
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise EnvironmentalValidationError(
                    f"invalid value for {parameter} on {raw_day}"
                ) from exc
            is_fill = numeric == fill_value
            values.append(
                EnvironmentalValue(
                    day=day,
                    parameter=parameter,
                    value=None if is_fill else numeric,
                    unit=unit,
                    is_fill_value=is_fill,
                )
            )
    expected_days = (expected_end - expected_start).days + 1
    if len(values) != expected_days * len(PARAMETERS):
        raise EnvironmentalValidationError(
            "POWER response has an incomplete or duplicate day/parameter grid"
        )
    digest = hashlib.sha256(body).hexdigest()
    snapshot_id = f"nasa_power_{retrieved_at:%Y%m%dT%H%M%SZ}_{digest[:12]}"
    return EnvironmentalReceipt(
        provider="NASA POWER",
        product="daily_point_v2",
        state=state,
        requested_url=requested_url,
        final_url=final_url,
        retrieved_at=retrieved_at,
        sha256=digest,
        byte_length=len(body),
        longitude=float(coordinates[0]),
        latitude=float(coordinates[1]),
        start=expected_start,
        end=expected_end,
        api_version=str(header.get("api", {}).get("version", "unknown")),
        time_standard="UTC",
        values=tuple(sorted(values, key=lambda item: (item.day, item.parameter))),
        warnings=(
            "coarse_grid_environmental_context_not_district_incidence",
            "fixed_demo_point_is_not_an_authoritative_district_centroid",
        ),
        source_snapshot_id=snapshot_id,
    )


def fetch_power_daily(
    *,
    longitude: float,
    latitude: float,
    start: date,
    end: date,
    policy: FetchPolicy | None = None,
) -> EnvironmentalReceipt:
    requested_url = build_power_url(longitude=longitude, latitude=latitude, start=start, end=end)
    result = fetch_url(
        requested_url,
        source_id="nasa_power_daily_point",
        allowed_hosts=(POWER_HOST,),
        policy=policy,
        access_path="provider_api",
    )
    return parse_power_daily(
        result.body,
        requested_url=requested_url,
        final_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        expected_longitude=longitude,
        expected_latitude=latitude,
        expected_start=start,
        expected_end=end,
    )
