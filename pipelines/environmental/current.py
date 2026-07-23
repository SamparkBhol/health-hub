"""Near-real-time NASA POWER climate for the 30 Odisha districts.

The historical cache under ``data/environment/power_daily`` stops at the end of
the modelling window.  This module fetches the *recent* daily record for the
same 30 representative points, from the same validated POWER endpoint, so that
issue-time environmental features can be formed for the present week.

POWER publishes with roughly a two-day latency: the final one or two days of a
request come back as the provider's fill value.  Those days are already dropped
by :func:`~packages.forecasting.climate.weekly_from_receipt`, which only keeps
ISO weeks with seven observed days, so the most recent partial week never enters
a feature window.  Nothing here is interpolated and nothing is back-filled.

A point sample of a coarse global reanalysis is environmental context for a
district.  It is not a district-average exposure and it is not disease data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from workers.ingestion.safe_fetch import FetchError, FetchPolicy, fetch_url

from .districts import DistrictPoint, load_district_points
from .models import AcquisitionState, EnvironmentalReceipt
from .nasa_power import POWER_HOST, build_power_url, parse_power_daily

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "environment" / "power_recent"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "1.0.0"

#: Enough trailing days to form a 12-week window plus the provider's latency and
#: a partial leading ISO week.
DEFAULT_LOOKBACK_DAYS = 130
#: POWER's own publication latency; the request stops here rather than asking for
#: days the provider is certain not to have.
PROVIDER_LATENCY_DAYS = 2


class RecentClimateError(RuntimeError):
    """The near-real-time climate window could not be retrieved or verified."""


@dataclass(frozen=True, slots=True)
class RecentVintage:
    district_id: str
    canonical_name: str
    longitude: float
    latitude: float
    start: date
    end: date
    sha256: str
    byte_length: int
    retrieved_at: datetime
    api_version: str
    requested_url: str
    object_name: str
    observed_days: int
    fill_days: int
    last_observed_day: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "district_id": self.district_id,
            "canonical_name": self.canonical_name,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "retrieved_at": self.retrieved_at.isoformat().replace("+00:00", "Z"),
            "api_version": self.api_version,
            "requested_url": self.requested_url,
            "object_name": self.object_name,
            "observed_days": self.observed_days,
            "fill_days": self.fill_days,
            "last_observed_day": self.last_observed_day,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> RecentVintage:
        return cls(
            district_id=str(value["district_id"]),
            canonical_name=str(value["canonical_name"]),
            longitude=float(str(value["longitude"])),
            latitude=float(str(value["latitude"])),
            start=date.fromisoformat(str(value["start"])),
            end=date.fromisoformat(str(value["end"])),
            sha256=str(value["sha256"]),
            byte_length=int(str(value["byte_length"])),
            retrieved_at=datetime.fromisoformat(str(value["retrieved_at"]).replace("Z", "+00:00")),
            api_version=str(value["api_version"]),
            requested_url=str(value["requested_url"]),
            object_name=str(value["object_name"]),
            observed_days=int(str(value["observed_days"])),
            fill_days=int(str(value["fill_days"])),
            last_observed_day=(
                None if value.get("last_observed_day") in (None, "None") else
                str(value["last_observed_day"])
            ),
        )


def default_window(today: date | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS):
    """The most recent window POWER can plausibly answer for, as (start, end)."""

    anchor = (today or datetime.now(UTC).date()) - timedelta(days=PROVIDER_LATENCY_DAYS)
    return anchor - timedelta(days=lookback_days - 1), anchor


def manifest_path(root: Path | None = None) -> Path:
    return (root or CACHE_ROOT) / MANIFEST_NAME


def read_manifest(root: Path | None = None) -> dict[str, RecentVintage]:
    path = manifest_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["district_id"]): RecentVintage.from_dict(item)
        for item in payload.get("vintages", [])
    }


def load_recent_body(vintage: RecentVintage, root: Path | None = None) -> bytes:
    path = (root or CACHE_ROOT) / vintage.object_name
    if not path.exists():
        raise RecentClimateError(f"recent vintage {vintage.object_name} is missing")
    import gzip

    body = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(body).hexdigest()
    if digest != vintage.sha256:
        raise RecentClimateError(
            f"recent vintage {vintage.object_name} digest {digest} does not match "
            f"the manifest {vintage.sha256}"
        )
    return body


def load_recent_receipt(
    vintage: RecentVintage, root: Path | None = None
) -> EnvironmentalReceipt:
    """Re-run the full POWER contract check against a cached recent vintage."""

    return parse_power_daily(
        load_recent_body(vintage, root),
        requested_url=vintage.requested_url,
        final_url=vintage.requested_url,
        retrieved_at=vintage.retrieved_at,
        expected_longitude=vintage.longitude,
        expected_latitude=vintage.latitude,
        expected_start=vintage.start,
        expected_end=vintage.end,
        state=AcquisitionState.RETRIEVED_AND_VALIDATED,
    )


def _summarise(receipt: EnvironmentalReceipt) -> tuple[int, int, str | None]:
    rain = [value for value in receipt.values if value.parameter == "PRECTOTCORR"]
    observed = [value for value in rain if not value.is_fill_value]
    fill = [value for value in rain if value.is_fill_value]
    last = max((value.day for value in observed), default=None)
    return len(observed), len(fill), last.isoformat() if last else None


def fetch_recent_window(
    point: DistrictPoint,
    *,
    start: date,
    end: date,
    policy: FetchPolicy | None = None,
    root: Path | None = None,
    store: bool = True,
) -> tuple[EnvironmentalReceipt, RecentVintage]:
    """Fetch and validate one district's recent daily POWER window."""

    import gzip

    requested_url = build_power_url(
        longitude=point.longitude, latitude=point.latitude, start=start, end=end
    )
    result = fetch_url(
        requested_url,
        source_id=f"nasa_power_recent_point:{point.district_id}",
        allowed_hosts=(POWER_HOST,),
        policy=policy,
        access_path="provider_api",
    )
    receipt = parse_power_daily(
        result.body,
        requested_url=requested_url,
        final_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        expected_longitude=point.longitude,
        expected_latitude=point.latitude,
        expected_start=start,
        expected_end=end,
    )
    observed, fill, last = _summarise(receipt)
    name = f"{point.district_id}_recent.json.gz"
    if store:
        directory = root or CACHE_ROOT
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(gzip.compress(result.body, compresslevel=9, mtime=0))
    vintage = RecentVintage(
        district_id=point.district_id,
        canonical_name=point.canonical_name,
        longitude=point.longitude,
        latitude=point.latitude,
        start=start,
        end=end,
        sha256=hashlib.sha256(result.body).hexdigest(),
        byte_length=len(result.body),
        retrieved_at=result.receipt.retrieved_at,
        api_version=receipt.api_version,
        requested_url=requested_url,
        object_name=name,
        observed_days=observed,
        fill_days=fill,
        last_observed_day=last,
    )
    return receipt, vintage


def refresh_recent_cache(
    *,
    start: date | None = None,
    end: date | None = None,
    points: Sequence[DistrictPoint] | None = None,
    policy: FetchPolicy | None = None,
    root: Path | None = None,
    on_progress=None,
) -> dict[str, object]:
    """Refresh the recent POWER window for every district and rewrite the manifest."""

    if (start is None) != (end is None):
        raise ValueError("pass both start and end, or neither")
    if start is None or end is None:
        start, end = default_window()
    selected = tuple(points) if points is not None else load_district_points()
    directory = root or CACHE_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    vintages: list[RecentVintage] = []
    failures: list[dict[str, str]] = []
    for point in selected:
        try:
            _, vintage = fetch_recent_window(
                point, start=start, end=end, policy=policy, root=root
            )
        except (FetchError, ValueError) as exc:
            failures.append(
                {
                    "district_id": point.district_id,
                    "code": getattr(exc, "code", exc.__class__.__name__),
                    "detail": str(exc)[:300],
                }
            )
            if on_progress:
                on_progress(point.district_id, "failed")
            continue
        vintages.append(vintage)
        if on_progress:
            on_progress(point.district_id, "fetched")
    ordered = sorted(vintages, key=lambda item: item.district_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provider": "NASA POWER",
        "product": "daily_point_v2_near_real_time",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "provider_latency_days": PROVIDER_LATENCY_DAYS,
        "districts_requested": len(selected),
        "districts_cached": len(ordered),
        "failures": failures,
        "vintages": [item.as_dict() for item in ordered],
    }
    manifest_path(root).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def recent_data_edge(root: Path | None = None) -> str | None:
    """The most recent fully observed POWER day common to every cached district."""

    manifest = read_manifest(root)
    if not manifest:
        return None
    edges = [item.last_observed_day for item in manifest.values() if item.last_observed_day]
    if len(edges) != len(manifest):
        return None
    return min(edges)
