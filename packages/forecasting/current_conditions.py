"""Present-day environmental conditions for all 30 Odisha districts.

The published reported-outbreak model stops where the EpiClim catalogue stops.
Every map it can draw is retrospective.  This module answers the question that
one cannot: *what are the environmental conditions across Odisha right now, and
how unusual are they for this district at this time of year?*

Two real, current sources are fused:

* **NASA POWER** near-real-time daily point series for the same 30
  representative points the model was trained on.  Keyless, roughly two days
  behind, and used to form exactly the environmental features the historical
  fit used, which is what makes the two comparable at all.
* **India Meteorological Department**, through the public products enumerated in
  :mod:`pipelines.environmental.imd`: five-day district warnings, district
  nowcast, AWS/ARG, SYNOP and METAR station observations, city-observatory
  records and the CAP alert feed.

WHAT THE OUTPUT IS
------------------
An experimental environmental risk-factor context layer. For each
district it publishes where current conditions sit in that district's own
2009-2022 distribution for the same part of the calendar, plus IMD's own
official warning state.  It is deliberately **not**:

* a case forecast,
* an outbreak probability,
* incidence, or
* anything with a case number in it.

Districts whose recent climate record is too gappy to form the feature windows
return a typed refusal rather than a score.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pipelines.environmental.current import (
    RecentClimateError,
    load_recent_receipt,
    read_manifest,
    recent_data_edge,
)
from pipelines.environmental.districts import load_district_points

from .climate import (
    EXTENDED_FEATURE_NAMES,
    FEATURE_NAMES,
    HUMID_DAY_PCT,
    MAXIMUM_TRAILING_WEEKS,
    REFERENCE_T2M_C,
    WET_DAY_MM,
    DistrictClimateFeatures,
    build_feature_index,
    load_weekly_panel,
)
from .suitability import (
    QUANTITY,
    QUANTITY_STATEMENT,
    SuitabilityArtefactMissing,
    SuitabilityModel,
    load_model,
)
from .target import week_start

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = REPO_ROOT / "data" / "environment" / "current_conditions.json"
DEFAULT_REMOTE_OBJECT_KEY = "environment/current-conditions/latest.json"
MAXIMUM_REMOTE_LAYER_BYTES = 5 * 1024 * 1024
SCHEMA_VERSION = "1.0.0"
LAYER_VERSION = "current-environmental-conditions-1.0.0"

#: A trailing window formed from fewer than this share of its calendar days is
#: refused. Rain accumulations over an incomplete window are underestimates, so
#: the tolerance is deliberately tight and always disclosed.
MINIMUM_DAY_COVERAGE = 0.94

#: The MERRA-2 grid NASA POWER resamples is 0.5 degrees latitude by 0.625
#: degrees longitude. Two representative points can therefore fall in the same
#: cell and receive identical climate, which is a property of the provider grid
#: rather than an error. It is published per district so it cannot be mistaken
#: for a duplicated row.
POWER_GRID_LATITUDE_DEGREES = 0.5
POWER_GRID_LONGITUDE_DEGREES = 0.625

#: Published block first, then the descriptive-only extensions.  The prefix
#: ordering is what lets the scorer slice the published block off the front.
ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + EXTENDED_FEATURE_NAMES

NOT_A_FORECAST_WARNING = (
    "This layer describes observed weather, not disease. It carries no case "
    "counts, no incidence and no outbreak probability. A high environmental "
    "feature-index percentile means the weak fitted weather index is unusual for "
    "this time of year by the standard of the district's own 2009-2022 record; "
    "it does not establish transmission suitability or mean an outbreak is expected."
)


class CurrentConditionsUnavailable(RuntimeError):
    """The inputs for a current-conditions layer are missing."""


@dataclass(frozen=True, slots=True)
class DailySeries:
    """The observed daily record for one district, gaps left as gaps."""

    district_id: str
    rain_mm: dict[date, float]
    t2m_c: dict[date, float]
    tmax_c: dict[date, float]
    tmin_c: dict[date, float]
    rh_pct: dict[date, float]

    @property
    def last_observed(self) -> date | None:
        return max(self.rain_mm) if self.rain_mm else None


def daily_series_from_receipt(district_id: str, receipt) -> DailySeries:
    columns: dict[str, dict[date, float]] = {
        "PRECTOTCORR": {},
        "T2M": {},
        "T2M_MAX": {},
        "T2M_MIN": {},
        "RH2M": {},
    }
    for value in receipt.values:
        if value.is_fill_value or value.value is None:
            continue
        if value.parameter in columns:
            columns[value.parameter][value.day] = float(value.value)
    return DailySeries(
        district_id=district_id,
        rain_mm=columns["PRECTOTCORR"],
        t2m_c=columns["T2M"],
        tmax_c=columns["T2M_MAX"],
        tmin_c=columns["T2M_MIN"],
        rh_pct=columns["RH2M"],
    )


def anchor_week(series: DailySeries) -> date | None:
    """The last ISO week complete for every required daily parameter."""

    required = (
        series.rain_mm,
        series.t2m_c,
        series.tmax_c,
        series.tmin_c,
        series.rh_pct,
    )
    if any(not column for column in required):
        return None
    # Start from the latest date that every parameter could possibly support.
    candidate = week_start(min(max(column) for column in required))
    for _ in range(8):
        days = [candidate + timedelta(days=offset) for offset in range(7)]
        if all(day in column for column in required for day in days):
            return candidate
        candidate -= timedelta(weeks=1)
    return None


def _window(series: dict[date, float], end: date, days: int) -> tuple[list[float], float]:
    """Values observed in the ``days`` calendar days ending at ``end`` inclusive."""

    collected = [
        series[day] for offset in range(days) if (day := end - timedelta(days=offset)) in series
    ]
    return collected, len(collected) / days


@dataclass(frozen=True, slots=True)
class RecentFeatures:
    district_id: str
    issue_week: date
    window_end: date
    iso_week: int
    values: tuple[float, ...]
    coverage: float
    observed_days: int
    expected_days: int
    parameter_day_coverage: dict[str, float]
    window_day_coverage: dict[str, float]
    status: str
    reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "district_id": self.district_id,
            "issue_week": self.issue_week.isoformat(),
            "window_end": self.window_end.isoformat(),
            "iso_week": self.iso_week,
            "features": dict(
                zip(ALL_FEATURE_NAMES, [round(v, 4) for v in self.values], strict=True)
            ),
            "features_used_for_scoring": list(FEATURE_NAMES),
            "features_descriptive_only": list(EXTENDED_FEATURE_NAMES),
            "day_coverage": round(self.coverage, 4),
            "observed_days": self.observed_days,
            "expected_days": self.expected_days,
            "parameter_day_coverage": {
                key: round(value, 4) for key, value in sorted(self.parameter_day_coverage.items())
            },
            "window_day_coverage": {
                key: round(value, 4) for key, value in sorted(self.window_day_coverage.items())
            },
            "status": self.status,
            "reason_code": self.reason_code,
        }


def build_recent_features(
    series: DailySeries, climatology: DistrictClimateFeatures
) -> RecentFeatures | None:
    """Form the model's environmental features from the recent daily record.

    Window definitions are identical to the training path: when every calendar
    day is observed, summing daily values over the trailing 7N days is the same
    arithmetic as summing N complete weekly aggregates.  Where the provider has
    gaps, the window is computed over the days that exist, the coverage is
    published, and a window below :data:`MINIMUM_DAY_COVERAGE` is refused rather
    than filled in.
    """

    anchor = anchor_week(series)
    if anchor is None:
        return None
    end = anchor + timedelta(days=6)
    horizon = MAXIMUM_TRAILING_WEEKS * 7
    iso_week = anchor.isocalendar().week

    rain1, rain1_coverage = _window(series.rain_mm, end, 7)
    rain2, rain2_coverage = _window(series.rain_mm, end, 14)
    rain4, rain4_coverage = _window(series.rain_mm, end, 28)
    rain8, rain8_coverage = _window(series.rain_mm, end, horizon)
    t2m4, t2m4_coverage = _window(series.t2m_c, end, 28)
    tmax4, tmax4_coverage = _window(series.tmax_c, end, 28)
    tmin4, tmin4_coverage = _window(series.tmin_c, end, 28)
    rh4, rh4_coverage = _window(series.rh_pct, end, 28)

    common_4w_days = [
        end - timedelta(days=offset)
        for offset in range(28)
        if all(
            end - timedelta(days=offset) in column
            for column in (
                series.rain_mm,
                series.t2m_c,
                series.tmax_c,
                series.tmin_c,
                series.rh_pct,
            )
        )
    ]
    common_4w_coverage = len(common_4w_days) / 28
    window_coverage = {
        "rain_1w": rain1_coverage,
        "rain_2w": rain2_coverage,
        "rain_4w": rain4_coverage,
        "rain_8w": rain8_coverage,
        "t2m_4w": t2m4_coverage,
        "tmax_4w": tmax4_coverage,
        "tmin_4w": tmin4_coverage,
        "rh_4w": rh4_coverage,
        "all_parameters_common_4w": common_4w_coverage,
    }
    parameter_coverage = {
        "rain_mm": min(
            rain1_coverage,
            rain2_coverage,
            rain4_coverage,
            rain8_coverage,
        ),
        "t2m_c": t2m4_coverage,
        "tmax_c": tmax4_coverage,
        "tmin_c": tmin4_coverage,
        "rh_pct": rh4_coverage,
    }
    expected_by_window = {
        "rain_1w": 7,
        "rain_2w": 14,
        "rain_4w": 28,
        "rain_8w": horizon,
        "t2m_4w": 28,
        "tmax_4w": 28,
        "tmin_4w": 28,
        "rh_4w": 28,
        "all_parameters_common_4w": 28,
    }
    observed_by_window = {
        "rain_1w": len(rain1),
        "rain_2w": len(rain2),
        "rain_4w": len(rain4),
        "rain_8w": len(rain8),
        "t2m_4w": len(t2m4),
        "tmax_4w": len(tmax4),
        "tmin_4w": len(tmin4),
        "rh_4w": len(rh4),
        "all_parameters_common_4w": len(common_4w_days),
    }
    # Prefer the longer window when several windows tie for minimum coverage,
    # preserving the historical 56/56 summary on complete input.
    limiting_window = min(
        window_coverage,
        key=lambda name: (window_coverage[name], -expected_by_window[name]),
    )
    coverage = window_coverage[limiting_window]
    observed = observed_by_window[limiting_window]
    expected = expected_by_window[limiting_window]

    def refusal(reason_code: str) -> RecentFeatures:
        return RecentFeatures(
            district_id=series.district_id,
            issue_week=anchor,
            window_end=end,
            iso_week=iso_week,
            values=(),
            coverage=coverage,
            observed_days=observed,
            expected_days=expected,
            parameter_day_coverage=parameter_coverage,
            window_day_coverage=window_coverage,
            status="insufficient_evidence",
            reason_code=reason_code,
        )

    if coverage < MINIMUM_DAY_COVERAGE:
        return refusal("RECENT_REQUIRED_PARAMETER_BELOW_MINIMUM_DAY_COVERAGE")

    rain1_sum = sum(rain1)
    rain2_sum = sum(rain2)
    rain4_sum = sum(rain4)
    rain8_sum = sum(rain8)
    t2m4_mean = sum(t2m4) / len(t2m4)
    tmax4_mean = sum(tmax4) / len(tmax4)
    rh4_mean = sum(rh4) / len(rh4)
    dtr4_mean = sum(series.tmax_c[day] - series.tmin_c[day] for day in common_4w_days) / len(
        common_4w_days
    )

    rain_anomaly = climatology.anomaly("rain4", anchor, rain4_sum / 4.0)
    rain8_anomaly = climatology.anomaly("rain8", anchor, rain8_sum / 8.0)
    t2m_anomaly = climatology.anomaly("t2m4", anchor, t2m4_mean)
    rh_anomaly = climatology.anomaly("rh4", anchor, rh4_mean)
    if None in (rain_anomaly, rain8_anomaly, t2m_anomaly, rh_anomaly):
        return refusal("NO_HISTORICAL_CLIMATOLOGY_FOR_THIS_ISO_WEEK")

    humid_days = sum(1 for value in rh4 if value >= HUMID_DAY_PCT)
    longest_dry = 0
    running = 0
    for offset in range(27, -1, -1):
        day = end - timedelta(days=offset)
        value = series.rain_mm.get(day)
        if value is None:
            continue
        if value < WET_DAY_MM:
            running += 1
            longest = running
            longest_dry = max(longest_dry, longest)
        else:
            running = 0

    values = (
        rain1_sum,
        rain2_sum,
        rain4_sum,
        rain8_sum,
        float(rain_anomaly),  # type: ignore[arg-type]
        t2m4_mean,
        float(t2m_anomaly),  # type: ignore[arg-type]
        rh4_mean,
        float(rh_anomaly),  # type: ignore[arg-type]
        dtr4_mean,
        float(rain8_anomaly),  # type: ignore[arg-type]
        float(longest_dry),
        float(humid_days),
        tmax4_mean,
        rain4_sum * (t2m4_mean - REFERENCE_T2M_C),
    )
    if len(values) != len(ALL_FEATURE_NAMES):  # pragma: no cover - guarded by tests
        raise RuntimeError(
            f"near-real-time feature width {len(values)} does not match the "
            f"expected {len(ALL_FEATURE_NAMES)}"
        )
    return RecentFeatures(
        district_id=series.district_id,
        issue_week=anchor,
        window_end=end,
        iso_week=iso_week,
        values=values,
        coverage=coverage,
        observed_days=observed,
        expected_days=expected,
        parameter_day_coverage=parameter_coverage,
        window_day_coverage=window_coverage,
        status="observed",
    )


def build_layer(
    *,
    imd_payload: dict[str, Any] | None = None,
    suitability: SuitabilityModel | None = None,
    recent_root: Path | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Assemble the 30-district current-conditions layer from real inputs."""

    manifest = read_manifest(recent_root)
    if not manifest:
        raise CurrentConditionsUnavailable(
            "no near-real-time NASA POWER cache; run "
            "`python scripts/collect_environment.py --mode current-conditions`"
        )
    try:
        model = suitability or load_model()
    except SuitabilityArtefactMissing as exc:
        raise CurrentConditionsUnavailable(str(exc)) from exc

    climatology = build_feature_index(load_weekly_panel())
    points = load_district_points()
    names = {point.district_id: point.canonical_name for point in points}
    cells: dict[tuple[int, int], list[str]] = {}
    for point in points:
        key = (
            round(point.latitude / POWER_GRID_LATITUDE_DEGREES),
            round(point.longitude / POWER_GRID_LONGITUDE_DEGREES),
        )
        cells.setdefault(key, []).append(point.district_id)
    shares_cell = {
        district_id: sorted(set(members) - {district_id})
        for members in cells.values()
        for district_id in members
        if len(members) > 1
    }

    imd_index: dict[str, dict[str, Any]] = {}
    if imd_payload is not None:
        from pipelines.environmental.imd import district_signal_index, merge_station_rainfall

        imd_index = district_signal_index(imd_payload)
        collected = str(imd_payload.get("collected_at", ""))[:10] or None
        for entry in imd_index.values():
            entry["imd_station_rainfall"] = merge_station_rainfall(
                entry.get("imd_station_observations", []), as_of=collected
            )

    districts: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for district_id, vintage in sorted(manifest.items()):
        row: dict[str, Any] = {
            "district_id": district_id,
            "canonical_name": names.get(district_id, vintage.canonical_name),
            "is_synthetic": False,
            "shares_climate_grid_cell_with": shares_cell.get(district_id, []),
        }
        try:
            receipt = load_recent_receipt(vintage, recent_root)
        except RecentClimateError as exc:
            row.update(
                {
                    "status": "insufficient_evidence",
                    "reason_code": "RECENT_CLIMATE_VINTAGE_UNREADABLE",
                    "detail": str(exc)[:200],
                }
            )
            districts.append(row)
            statuses["insufficient_evidence"] += 1
            continue
        series = daily_series_from_receipt(district_id, receipt)
        features = build_recent_features(series, climatology[district_id])
        if features is None:
            row.update(
                {
                    "status": "insufficient_evidence",
                    "reason_code": "NO_COMPLETE_RECENT_ISO_WEEK",
                }
            )
            districts.append(row)
            statuses["insufficient_evidence"] += 1
        elif features.status != "observed":
            row.update(
                {
                    "status": "insufficient_evidence",
                    "reason_code": features.reason_code,
                    "environment": features.as_dict(),
                }
            )
            districts.append(row)
            statuses["insufficient_evidence"] += 1
        else:
            # The suitability model scores on the published block only; the
            # extended columns are carried for description, never for scoring.
            score = model.score(
                district_id, features.iso_week, features.values[: len(FEATURE_NAMES)]
            )
            row.update(
                {
                    "status": "observed" if score.status == "scored" else "partial",
                    "reason_code": score.reason_code,
                    "environment": features.as_dict(),
                    "suitability": score.as_dict(),
                }
            )
            districts.append(row)
            statuses[str(row["status"])] += 1
        row.update(imd_index.get(district_id, {}))
        row.pop("district_id", None)
        row["district_id"] = district_id

    edge = recent_data_edge(recent_root)
    scored = [
        item
        for item in districts
        if item.get("suitability", {}).get("suitability_percentile") is not None
    ]
    ranked = sorted(
        scored,
        key=lambda item: float(item["suitability"]["suitability_percentile"]),
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer_version": LAYER_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "as_of": (today or datetime.now(UTC).date()).isoformat(),
        "is_synthetic": False,
        "uses_real_odisha_data": True,
        "quantity": {
            "kind": QUANTITY,
            "statement": QUANTITY_STATEMENT,
            "is_incidence": False,
            "is_case_count": False,
            "is_outbreak_probability": False,
            "is_forecast": False,
        },
        "coverage": {
            "districts": len(districts),
            "scored": len(scored),
            "status_counts": dict(sorted(statuses.items())),
        },
        "data_edge": {
            "nasa_power_last_observed_day": edge,
            "nasa_power_window": f"{manifest[next(iter(sorted(manifest)))].start.isoformat()}"
            f"..{manifest[next(iter(sorted(manifest)))].end.isoformat()}",
            "imd_collected_at": (imd_payload or {}).get("collected_at"),
        },
        "sources": {
            "climate": {
                "provider": "NASA POWER",
                "product": "daily_point_v2_near_real_time",
                "access": "keyless public API",
                "districts": len(manifest),
                "grid_degrees": [
                    POWER_GRID_LATITUDE_DEGREES,
                    POWER_GRID_LONGITUDE_DEGREES,
                ],
                "distinct_grid_cells": len(cells),
                "districts_sharing_a_grid_cell": sorted(shares_cell),
                "grid_note": (
                    "Districts sharing a provider grid cell receive identical "
                    "climate values. That is the resolution of the reanalysis, "
                    "not a duplicated record."
                ),
            },
            "meteorology": {
                "provider": "India Meteorological Department",
                "products": sorted((imd_payload or {}).get("products", {})),
                "blocked_surfaces": (imd_payload or {}).get("blocked_surfaces", []),
                "failures": (imd_payload or {}).get("failures", []),
            },
            "suitability_model": {
                "model_version": model.payload.get("model_version"),
                "generated_at": model.generated_at,
                "fitted_against": model.payload.get("fitted_against"),
            },
        },
        "ranking_most_unusual_conditions": [
            {
                "district_id": item["district_id"],
                "canonical_name": item["canonical_name"],
                "suitability_percentile": item["suitability"]["suitability_percentile"],
                "band": item["suitability"]["band"],
            }
            for item in ranked[:10]
        ],
        "districts": districts,
        "warnings": [
            NOT_A_FORECAST_WARNING,
            (
                "NASA POWER is a coarse global reanalysis sampled at one "
                "representative interior point per district. It is environmental "
                "context, not a district-average exposure."
            ),
            (
                "IMD warning colours and hazard codes are IMD's own official "
                "meteorological products, reproduced verbatim. They are weather "
                "warnings, not health warnings."
            ),
            # The fitted model's own caveats travel with every rendering of its
            # output, not just with the artefact a reviewer might never open.
            *[str(item) for item in model.payload.get("warnings", [])],
        ],
    }


def write_layer(payload: dict[str, Any], path: Path | None = None) -> Path:
    target = path or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def current_conditions_layer(path: Path | None = None) -> dict[str, Any]:
    """Alias used by the API layer; identical to :func:`load_layer`."""

    return load_layer(path)


def current_conditions_map(path: Path | None = None) -> dict[str, Any]:
    """Alias used by the API layer; identical to :func:`map_payload`."""

    return map_payload(path)


def _remote_configuration() -> tuple[str, str, str, str, str] | None:
    values = (
        os.getenv("R2_ENDPOINT_URL", "").strip(),
        os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        os.getenv("R2_BUCKET", "").strip(),
        os.getenv("CURRENT_CONDITIONS_R2_KEY", DEFAULT_REMOTE_OBJECT_KEY).strip(),
    )
    # Remote reads are opt-in through a complete R2 configuration. A partially
    # configured deployment fails explicitly instead of silently serving the
    # bundled snapshot as though it were current.
    configured = [bool(value) for value in values[:4]]
    if not any(configured):
        return None
    if not all(configured) or not values[4]:
        raise CurrentConditionsUnavailable("current-conditions R2 configuration is incomplete")
    return values


def _load_remote_layer(configuration: tuple[str, str, str, str, str]) -> dict[str, Any]:
    endpoint, access_key, secret_key, bucket, key = configuration
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(
                connect_timeout=3,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read(MAXIMUM_REMOTE_LAYER_BYTES + 1)
    except Exception as exc:  # provider/client exceptions share no stable base
        raise CurrentConditionsUnavailable(
            f"current-conditions remote layer unavailable at r2://{bucket}/{key}: {exc}"
        ) from exc
    if len(body) > MAXIMUM_REMOTE_LAYER_BYTES:
        raise CurrentConditionsUnavailable("current-conditions remote layer is too large")
    expected_digest = (response.get("Metadata") or {}).get("content-sha256")
    actual_digest = hashlib.sha256(body).hexdigest()
    if expected_digest and expected_digest != actual_digest:
        raise CurrentConditionsUnavailable(
            "current-conditions remote layer failed its content digest check"
        )
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentConditionsUnavailable(
            f"current-conditions remote layer is not valid UTF-8 JSON: {exc}"
        ) from exc


def _validate_layer(payload: dict[str, Any]) -> dict[str, Any]:
    """Enforce the non-forecast contract for local and remote artefacts alike."""

    if payload.get("is_synthetic") is not False:
        raise CurrentConditionsUnavailable(
            "current-conditions layer must declare is_synthetic=false"
        )
    quantity = payload.get("quantity", {})
    if quantity.get("is_case_count") is not False or quantity.get("is_incidence") is not False:
        raise CurrentConditionsUnavailable(
            "current-conditions layer must declare that it is neither incidence nor a case count"
        )
    if (
        quantity.get("is_outbreak_probability") is not False
        or quantity.get("is_forecast") is not False
    ):
        raise CurrentConditionsUnavailable(
            "current-conditions layer must declare that it is neither an outbreak "
            "probability nor a disease forecast"
        )
    if not isinstance(payload.get("districts"), list) or len(payload["districts"]) != 30:
        raise CurrentConditionsUnavailable(
            "current-conditions layer must contain exactly 30 district rows"
        )
    return payload


def load_layer(path: Path | None = None) -> dict[str, Any]:
    if path is None and (configuration := _remote_configuration()) is not None:
        return _validate_layer(_load_remote_layer(configuration))
    target = path or ARTEFACT_PATH
    if not target.exists():
        raise CurrentConditionsUnavailable(
            f"no current-conditions layer at {target}; run "
            "`python scripts/collect_environment.py --mode current-conditions`"
        )
    try:
        payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CurrentConditionsUnavailable(
            f"current-conditions layer is not valid JSON: {exc}"
        ) from exc
    return _validate_layer(payload)


def map_payload(path: Path | None = None) -> dict[str, Any]:
    """A compact, render-ready view: one row per district, no raw feature dump."""

    payload = load_layer(path)
    rows: list[dict[str, Any]] = []
    for district in payload["districts"]:
        suitability = district.get("suitability") or {}
        environment = district.get("environment") or {}
        features = environment.get("features") or {}
        warning = district.get("imd_warning") or {}
        nowcast = district.get("imd_nowcast") or {}
        city = district.get("imd_city_observation") or {}
        rows.append(
            {
                "district_id": district["district_id"],
                "canonical_name": district["canonical_name"],
                "status": district.get("status"),
                "reason_code": district.get("reason_code"),
                "is_synthetic": False,
                "suitability_percentile": suitability.get("suitability_percentile"),
                "band": suitability.get("band"),
                "top_drivers": suitability.get("top_drivers", [])[:3],
                "environment_week": environment.get("issue_week"),
                "rain_4w_mm": features.get("rain_4w_mm"),
                "rain_4w_anomaly_sd": features.get("rain_4w_anomaly_sd"),
                "rh_4w_pct": features.get("rh_4w_pct"),
                "t2m_4w_c": features.get("t2m_4w_c"),
                "longest_dry_run_4w_days": features.get("longest_dry_run_4w_days"),
                "imd_peak_warning_next_5_days": warning.get("peak_severity_next_5_days"),
                "imd_warning_days": warning.get("days", []),
                "imd_nowcast_severity": nowcast.get("severity"),
                "imd_nowcast_valid_upto_ist": nowcast.get("valid_upto_ist"),
                "imd_rainfall_24h_mm": city.get("rainfall_24h_mm"),
                "imd_station": city.get("station_name"),
                "imd_station_rainfall": district.get("imd_station_rainfall"),
            }
        )
    rows.sort(key=lambda item: str(item["district_id"]))
    return {
        "schema_version": payload["schema_version"],
        "layer_version": payload["layer_version"],
        "generated_at": payload["generated_at"],
        "as_of": payload["as_of"],
        "is_synthetic": False,
        "quantity": payload["quantity"],
        "coverage": payload["coverage"],
        "data_edge": payload["data_edge"],
        "ranking_most_unusual_conditions": payload["ranking_most_unusual_conditions"],
        "districts": rows,
        "warnings": payload["warnings"],
    }
