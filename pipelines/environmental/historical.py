"""Historical NASA POWER daily climate cache for the 30 Odisha districts.

Every cached object is the *verbatim* provider response body, gzipped, with a
manifest that records the SHA-256 of the uncompressed bytes.  Reads re-run the
full :func:`parse_power_daily` contract check, so a corrupted or silently
re-issued vintage fails closed rather than entering the model.

NASA POWER is a coarse global reanalysis.  A point sample is environmental
context for a district; it is not a district-average exposure and it is not
disease surveillance.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from workers.ingestion.safe_fetch import FetchError, FetchPolicy, fetch_url

from .districts import POINT_WARNINGS, DistrictPoint, load_district_points
from .models import AcquisitionState, EnvironmentalReceipt
from .nasa_power import POWER_HOST, build_power_url, parse_power_daily

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "environment" / "power_daily"
MANIFEST_NAME = "manifest.json"
CACHE_SCHEMA_VERSION = "1.0.0"


class ClimateCacheError(RuntimeError):
    """The on-disk climate cache is missing or does not match its manifest."""


@dataclass(frozen=True, slots=True)
class CachedVintage:
    district_id: str
    canonical_name: str
    longitude: float
    latitude: float
    point_method: str
    start: date
    end: date
    sha256: str
    byte_length: int
    retrieved_at: datetime
    api_version: str
    requested_url: str
    object_name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "district_id": self.district_id,
            "canonical_name": self.canonical_name,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "point_method": self.point_method,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "retrieved_at": self.retrieved_at.isoformat().replace("+00:00", "Z"),
            "api_version": self.api_version,
            "requested_url": self.requested_url,
            "object_name": self.object_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CachedVintage:
        return cls(
            district_id=str(value["district_id"]),
            canonical_name=str(value["canonical_name"]),
            longitude=float(str(value["longitude"])),
            latitude=float(str(value["latitude"])),
            point_method=str(value["point_method"]),
            start=date.fromisoformat(str(value["start"])),
            end=date.fromisoformat(str(value["end"])),
            sha256=str(value["sha256"]),
            byte_length=int(str(value["byte_length"])),
            retrieved_at=datetime.fromisoformat(str(value["retrieved_at"]).replace("Z", "+00:00")),
            api_version=str(value["api_version"]),
            requested_url=str(value["requested_url"]),
            object_name=str(value["object_name"]),
        )


def object_name(district_id: str, start: date, end: date) -> str:
    return f"{district_id}_{start:%Y%m%d}_{end:%Y%m%d}.json.gz"


def manifest_path(root: Path | None = None) -> Path:
    return (root or CACHE_ROOT) / MANIFEST_NAME


def read_manifest(root: Path | None = None) -> dict[str, CachedVintage]:
    path = manifest_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["district_id"]): CachedVintage.from_dict(item)
        for item in payload.get("vintages", [])
    }


def write_manifest(vintages: Iterable[CachedVintage], root: Path | None = None) -> Path:
    directory = root or CACHE_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(vintages, key=lambda item: item.district_id)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "provider": "NASA POWER",
        "product": "daily_point_v2",
        "parameters": ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"],
        "warnings": list(POINT_WARNINGS),
        "vintage_count": len(ordered),
        "vintages": [item.as_dict() for item in ordered],
    }
    path = manifest_path(directory)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def store_body(
    body: bytes,
    *,
    point: DistrictPoint,
    start: date,
    end: date,
    retrieved_at: datetime,
    requested_url: str,
    api_version: str,
    root: Path | None = None,
) -> CachedVintage:
    directory = root or CACHE_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    name = object_name(point.district_id, start, end)
    # mtime=0 keeps the gzip container byte-reproducible; provenance lives in
    # the manifest digest of the *uncompressed* provider body.
    (directory / name).write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    return CachedVintage(
        district_id=point.district_id,
        canonical_name=point.canonical_name,
        longitude=point.longitude,
        latitude=point.latitude,
        point_method=point.method,
        start=start,
        end=end,
        sha256=hashlib.sha256(body).hexdigest(),
        byte_length=len(body),
        retrieved_at=retrieved_at,
        api_version=api_version,
        requested_url=requested_url,
        object_name=name,
    )


def load_receipt(vintage: CachedVintage, root: Path | None = None) -> EnvironmentalReceipt:
    """Re-validate a cached body against the POWER response contract."""

    directory = root or CACHE_ROOT
    path = directory / vintage.object_name
    if not path.exists():
        raise ClimateCacheError(f"cached vintage {vintage.object_name} is missing")
    body = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(body).hexdigest()
    if digest != vintage.sha256:
        raise ClimateCacheError(
            f"cached vintage {vintage.object_name} digest {digest} "
            f"does not match manifest {vintage.sha256}"
        )
    return parse_power_daily(
        body,
        requested_url=vintage.requested_url,
        final_url=vintage.requested_url,
        retrieved_at=vintage.retrieved_at,
        expected_longitude=vintage.longitude,
        expected_latitude=vintage.latitude,
        expected_start=vintage.start,
        expected_end=vintage.end,
        state=AcquisitionState.RETRIEVED_AND_VALIDATED,
    )


def fetch_district_vintage(
    point: DistrictPoint,
    *,
    start: date,
    end: date,
    policy: FetchPolicy | None = None,
    root: Path | None = None,
) -> CachedVintage:
    requested_url = build_power_url(
        longitude=point.longitude, latitude=point.latitude, start=start, end=end
    )
    result = fetch_url(
        requested_url,
        source_id=f"nasa_power_daily_point:{point.district_id}",
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
    return store_body(
        result.body,
        point=point,
        start=start,
        end=end,
        retrieved_at=result.receipt.retrieved_at,
        requested_url=requested_url,
        api_version=receipt.api_version,
        root=root,
    )


def build_cache(
    *,
    start: date,
    end: date,
    points: Sequence[DistrictPoint] | None = None,
    policy: FetchPolicy | None = None,
    root: Path | None = None,
    refresh: bool = False,
    on_progress=None,
) -> dict[str, object]:
    """Fetch (or reuse) one daily vintage per district and rewrite the manifest."""

    selected = tuple(points) if points is not None else load_district_points()
    existing = read_manifest(root)
    directory = root or CACHE_ROOT
    vintages: list[CachedVintage] = []
    failures: list[dict[str, str]] = []
    fetched = 0
    reused = 0
    for point in selected:
        cached = existing.get(point.district_id)
        reusable = (
            cached is not None
            and not refresh
            and cached.start == start
            and cached.end == end
            and (directory / cached.object_name).exists()
        )
        if reusable and cached is not None:
            vintages.append(cached)
            reused += 1
            if on_progress:
                on_progress(point.district_id, "reused")
            continue
        try:
            vintage = fetch_district_vintage(
                point, start=start, end=end, policy=policy, root=root
            )
        except (FetchError, ValueError) as exc:
            failures.append(
                {
                    "district_id": point.district_id,
                    "code": getattr(exc, "code", exc.__class__.__name__),
                    "detail": str(exc),
                }
            )
            if on_progress:
                on_progress(point.district_id, "failed")
            continue
        vintages.append(vintage)
        fetched += 1
        if on_progress:
            on_progress(point.district_id, "fetched")
    write_manifest(vintages, root)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "districts_requested": len(selected),
        "districts_cached": len(vintages),
        "fetched": fetched,
        "reused": reused,
        "failures": failures,
        "manifest": str(manifest_path(root)),
    }
