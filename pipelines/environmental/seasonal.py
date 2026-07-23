"""ECMWF EC46/SEAS5 ensemble outlooks through Open-Meteo.

The provider returns 51 ensemble members.  We retain only compact 30-day lead
window summaries, uncertainty intervals, the exact request URL and a digest of
the source response.  This is future environmental context at roughly 36 km;
it is not a district weather forecast or disease probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .districts import DistrictPoint, load_district_points

API_ORIGIN = "https://seasonal-api.open-meteo.com"
API_PATH = "/v1/seasonal"
MODEL = "ecmwf_seasonal_seamless"
FORECAST_DAYS = 120
ENSEMBLE_MEMBERS = 51
WINDOW_DAYS = 30
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "environment" / "seasonal_outlook.json"


class SeasonalForecastError(RuntimeError):
    """The seasonal forecast response failed validation."""


@dataclass(frozen=True, slots=True)
class SeasonalWindow:
    horizon_month: int
    start_date: str
    end_date: str
    ensemble_size: int
    precipitation_mean_mm: float
    precipitation_p10_mm: float
    precipitation_p90_mm: float
    temperature_mean_c: float
    temperature_p10_c: float
    temperature_p90_c: float


@dataclass(frozen=True, slots=True)
class DistrictSeasonalOutlook:
    district_id: str
    district_name: str
    requested_latitude: float
    requested_longitude: float
    grid_latitude: float
    grid_longitude: float
    model: str
    source_url: str
    source_sha256: str
    windows: tuple[SeasonalWindow, ...]


def build_url(point: DistrictPoint) -> str:
    query = urlencode(
        {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "daily": "temperature_2m_mean,precipitation_sum",
            "models": MODEL,
            "forecast_days": FORECAST_DAYS,
            "timezone": "Asia/Kolkata",
        }
    )
    return f"{API_ORIGIN}{API_PATH}?{query}"


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_seasonal_response(
    point: DistrictPoint, body: bytes, *, source_url: str
) -> DistrictSeasonalOutlook:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise SeasonalForecastError("seasonal provider returned invalid JSON") from error
    if payload.get("error"):
        raise SeasonalForecastError(str(payload.get("reason", "seasonal provider error")))
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        raise SeasonalForecastError("seasonal response has no daily object")
    times = daily.get("time")
    if not isinstance(times, list) or len(times) < WINDOW_DAYS * 3:
        raise SeasonalForecastError("seasonal response is shorter than three 30-day windows")

    rain_members: list[list[float]] = []
    temperature_members: list[list[float]] = []
    # The API names the control member without a suffix and the 50 perturbed
    # members ``member01`` through ``member50``: 51 trajectories in total.
    for member in range(ENSEMBLE_MEMBERS):
        suffix = "" if member == 0 else f"_member{member:02d}"
        rain = daily.get(f"precipitation_sum{suffix}")
        temperature = daily.get(f"temperature_2m_mean{suffix}")
        if not isinstance(rain, list) or not isinstance(temperature, list):
            raise SeasonalForecastError(f"seasonal response lacks ensemble member {member:02d}")
        if len(rain) != len(times) or len(temperature) != len(times):
            raise SeasonalForecastError(f"ensemble member {member:02d} length mismatch")
        try:
            rain_members.append([float(value) for value in rain])
            temperature_members.append([float(value) for value in temperature])
        except (TypeError, ValueError) as error:
            raise SeasonalForecastError(
                f"ensemble member {member:02d} contains a missing daily value"
            ) from error

    windows: list[SeasonalWindow] = []
    for horizon in range(1, 4):
        start = (horizon - 1) * WINDOW_DAYS
        end = horizon * WINDOW_DAYS
        rain_totals = [sum(values[start:end]) for values in rain_members]
        temperature_means = [
            sum(values[start:end]) / WINDOW_DAYS for values in temperature_members
        ]
        windows.append(
            SeasonalWindow(
                horizon_month=horizon,
                start_date=str(times[start]),
                end_date=str(times[end - 1]),
                ensemble_size=ENSEMBLE_MEMBERS,
                precipitation_mean_mm=round(sum(rain_totals) / ENSEMBLE_MEMBERS, 3),
                precipitation_p10_mm=round(_quantile(rain_totals, 0.10), 3),
                precipitation_p90_mm=round(_quantile(rain_totals, 0.90), 3),
                temperature_mean_c=round(
                    sum(temperature_means) / ENSEMBLE_MEMBERS, 3
                ),
                temperature_p10_c=round(_quantile(temperature_means, 0.10), 3),
                temperature_p90_c=round(_quantile(temperature_means, 0.90), 3),
            )
        )
    return DistrictSeasonalOutlook(
        district_id=point.district_id,
        district_name=point.canonical_name,
        requested_latitude=point.latitude,
        requested_longitude=point.longitude,
        grid_latitude=float(payload["latitude"]),
        grid_longitude=float(payload["longitude"]),
        model=MODEL,
        source_url=source_url,
        source_sha256=hashlib.sha256(body).hexdigest(),
        windows=tuple(windows),
    )


def fetch_district_outlook(point: DistrictPoint) -> DistrictSeasonalOutlook:
    url = build_url(point)
    response = httpx.get(url, timeout=httpx.Timeout(45, connect=10), follow_redirects=True)
    response.raise_for_status()
    return parse_seasonal_response(point, response.content, source_url=url)


def refresh_seasonal_outlook(
    *,
    points: tuple[DistrictPoint, ...] | None = None,
    destination: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    selected = points or load_district_points()
    with ThreadPoolExecutor(max_workers=6) as pool:
        districts = list(pool.map(fetch_district_outlook, selected))
    if len(districts) != 30 or len({item.district_id for item in districts}) != 30:
        raise SeasonalForecastError("seasonal collection did not return all 30 districts")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": generated_at,
        "provider": "Open-Meteo seasonal API",
        "underlying_models": "ECMWF EC46 for the first 46 days, then ECMWF SEAS5",
        "model": MODEL,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "spatial_resolution": "approximately 36 km; representative point, not district average",
        "bias_correction": "none",
        "license": "ECMWF/Open-Meteo attribution applies; see SOURCES.md",
        "districts": [
            asdict(item) for item in sorted(districts, key=lambda item: item.district_id)
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def load_seasonal_outlook(path: Path = OUTPUT_PATH) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
