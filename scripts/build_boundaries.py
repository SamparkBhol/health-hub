#!/usr/bin/env python3
"""Build the pinned Odisha district demo GeoJSON from DataMeet Census 2011.

This is a reproducibility tool, not a runtime network dependency.  It verifies
every downloaded shapefile component before reading it and refuses to emit a
map unless the Odisha subset contains exactly the frozen 30-district set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import shapefile

COMMIT = "b3fbbde595310b397a55d718e0958ce249a4fa1f"
BASE_URL = (
    "https://raw.githubusercontent.com/datameet/maps/"
    f"{COMMIT}/Districts/Census_2011/2011_Dist"
)
SOURCE_HASHES = {
    "dbf": "2c7b03578fdf41a4c4e29941a564fa90bf0b0e4f865f5e54481ef7ac8494eb7d",
    "prj": "a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c4",
    "shp": "3636e627a519f67b4615b6a97df48c2b279e8c7b2566d3bfbc3eda1d06e33206",
    "shx": "12b6028dfac9e3f20686b3d0a44f9d3989acfdb3f3cda853881d1cf2d8579eab",
}
MAX_COMPONENT_BYTES = 20 * 1024 * 1024

SOURCE_TO_CANONICAL = {
    "Anugul": "Angul",
    "Baleshwar": "Balasore",
    "Bauda": "Boudh",
    "Debagarh": "Deogarh",
    "Jajapur": "Jajpur",
    "Jagatsinghapur": "Jagatsinghpur",
    "Kendujhar": "Keonjhar",
    "Nabarangapur": "Nabarangpur",
}

EXPECTED = {
    "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh",
    "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam",
    "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal",
    "Kendrapara", "Keonjhar", "Khordha", "Koraput", "Malkangiri",
    "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri",
    "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for extension, expected_hash in SOURCE_HASHES.items():
        target = source_dir / f"2011_Dist.{extension}"
        if not target.exists():
            request = urllib.request.Request(  # noqa: S310 - pinned HTTPS base URL
                f"{BASE_URL}.{extension}",
                headers={"User-Agent": "OdishaHealthHub-BoundaryBuilder/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_COMPONENT_BYTES:
                    raise SystemExit(
                        f"source component exceeds 20 MiB: {target.name}"
                    )
                total = 0
                with target.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_COMPONENT_BYTES:
                            raise SystemExit(
                                f"source component exceeds 20 MiB: {target.name}"
                            )
                        handle.write(chunk)
        actual = sha256(target)
        if actual != expected_hash:
            raise SystemExit(
                f"source hash mismatch for {target.name}: {actual} != {expected_hash}"
            )


def feature(record: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    source_name = str(record["DISTRICT"]).strip()
    canonical = SOURCE_TO_CANONICAL.get(source_name, source_name)
    slug = canonical.lower().replace(" ", "-")
    return {
        "type": "Feature",
        "id": f"OD-DIST-{slug}",
        "properties": {
            "district_id": f"OD-DIST-{slug}",
            "canonical_name": canonical,
            "source_name": source_name,
            "state_census_code": int(record["ST_CEN_CD"]),
            "district_census_code": int(record["DT_CEN_CD"]),
            "censuscode": int(record["censuscode"]),
            "boundary_vintage": "Census 2011",
            "boundary_authority": "community_demo",
        },
        "geometry": geometry,
    }


def build(source_dir: Path, output: Path, manifest: Path) -> None:
    acquire(source_dir)
    reader = shapefile.Reader(str(source_dir / "2011_Dist.shp"), encoding="latin1")
    features = []
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        if str(record.get("ST_NM", "")).strip() != "Odisha":
            continue
        features.append(feature(record, shape_record.shape.__geo_interface__))
    features.sort(key=lambda item: item["properties"]["canonical_name"])
    names = {item["properties"]["canonical_name"] for item in features}
    if len(features) != 30 or names != EXPECTED:
        raise SystemExit(
            f"expected frozen 30 Odisha districts; got {len(features)}; "
            f"missing={sorted(EXPECTED - names)} extra={sorted(names - EXPECTED)}"
        )

    collection = {
        "type": "FeatureCollection",
        "name": "odisha_districts_census_2011_demo",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": "1.0.0",
        "asset": output.name,
        "asset_sha256": sha256(output),
        "feature_count": len(features),
        "source_repository": "https://github.com/datameet/maps",
        "source_commit": COMMIT,
        "source_path": "Districts/Census_2011/2011_Dist.*",
        "source_sha256": SOURCE_HASHES,
        "source_vintage": "Census 2011",
        "licence": "CC-BY-2.5-IN",
        "attribution": "DataMeet India community — India District Map",
        "geometry_authority": "community_demo",
        "allowed_use": "evidence/coverage demonstration; not an operational State boundary",
    }
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/boundaries/odisha_districts_census_2011.geojson"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/boundaries/manifest.json")
    )
    args = parser.parse_args()
    if args.source_dir:
        build(args.source_dir, args.output, args.manifest)
        return
    with tempfile.TemporaryDirectory(prefix="odisha-boundaries-") as directory:
        build(Path(directory), args.output, args.manifest)


if __name__ == "__main__":
    main()
