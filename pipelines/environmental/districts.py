"""District representative points for environmental point queries.

The points are derived from the bundled Census-2011 community demo boundary and
are used *only* to choose where to sample a coarse global reanalysis grid.  They
are not administrative headquarters, not authoritative centroids, and the value
sampled at the point is not a district-average exposure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BOUNDARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "boundaries"
    / "odisha_districts_census_2011.geojson"
)

POINT_WARNINGS = (
    "representative_interior_point_not_administrative_headquarter",
    "point_sample_of_coarse_reanalysis_grid_not_district_average_exposure",
    "community_demo_boundary_not_authoritative_state_boundary",
)


@dataclass(frozen=True, slots=True)
class DistrictPoint:
    district_id: str
    canonical_name: str
    longitude: float
    latitude: float
    method: str


def _rings(geometry: dict) -> list[list[tuple[float, float]]]:
    kind = geometry["type"]
    if kind == "Polygon":
        polygons = [geometry["coordinates"]]
    elif kind == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:  # pragma: no cover - the bundled asset only carries the two types
        raise ValueError(f"unsupported geometry type {kind!r}")
    return [[(float(x), float(y)) for x, y in polygon[0]] for polygon in polygons]


def _ring_area(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _ring_centroid(ring: list[tuple[float, float]]) -> tuple[float, float, float]:
    area = _ring_area(ring)
    if abs(area) < 1e-12:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys), 0.0
    cx = 0.0
    cy = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return cx / (6.0 * area), cy / (6.0 * area), abs(area)


def _point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing > x:
                inside = not inside
    return inside


def _segment_distance(x: float, y: float, ring: list[tuple[float, float]]) -> float:
    best = float("inf")
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        dx = x2 - x1
        dy = y2 - y1
        length = dx * dx + dy * dy
        if length <= 0:
            distance = (x - x1) ** 2 + (y - y1) ** 2
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length))
            distance = (x - x1 - t * dx) ** 2 + (y - y1 - t * dy) ** 2
        best = min(best, distance)
    return best**0.5


def _representative_point(rings: list[list[tuple[float, float]]]) -> tuple[float, float, str]:
    """Return an interior point: the area centroid when it is inside, else a grid probe."""

    ranked = sorted(rings, key=lambda ring: abs(_ring_area(ring)), reverse=True)
    largest = ranked[0]
    cx, cy, _ = _ring_centroid(largest)
    if _point_in_ring(cx, cy, largest):
        return round(cx, 6), round(cy, 6), "area_centroid_inside_largest_ring"
    xs = [point[0] for point in largest]
    ys = [point[1] for point in largest]
    steps = 48
    best: tuple[float, float, float] | None = None
    for i in range(1, steps):
        px = min(xs) + (max(xs) - min(xs)) * i / steps
        for j in range(1, steps):
            py = min(ys) + (max(ys) - min(ys)) * j / steps
            if not _point_in_ring(px, py, largest):
                continue
            clearance = _segment_distance(px, py, largest)
            if best is None or clearance > best[2]:
                best = (px, py, clearance)
    if best is None:  # pragma: no cover - degenerate ring
        return round(cx, 6), round(cy, 6), "area_centroid_outside_ring_no_interior_probe"
    return round(best[0], 6), round(best[1], 6), "max_clearance_interior_grid_probe"


@lru_cache(maxsize=1)
def load_district_points(path: str | None = None) -> tuple[DistrictPoint, ...]:
    boundary = Path(path) if path else BOUNDARY_PATH
    payload = json.loads(boundary.read_text(encoding="utf-8"))
    points: list[DistrictPoint] = []
    for feature in payload["features"]:
        properties = feature["properties"]
        longitude, latitude, method = _representative_point(_rings(feature["geometry"]))
        points.append(
            DistrictPoint(
                district_id=str(properties["district_id"]),
                canonical_name=str(properties["canonical_name"]),
                longitude=longitude,
                latitude=latitude,
                method=method,
            )
        )
    points.sort(key=lambda item: item.district_id)
    return tuple(points)


@lru_cache(maxsize=1)
def _district_rings(path: str | None = None) -> tuple[tuple[str, str, tuple, tuple], ...]:
    """(district_id, canonical_name, outer rings, bounding box) for containment tests."""

    boundary = Path(path) if path else BOUNDARY_PATH
    payload = json.loads(boundary.read_text(encoding="utf-8"))
    entries: list[tuple[str, str, tuple, tuple]] = []
    for feature in payload["features"]:
        properties = feature["properties"]
        rings = tuple(tuple(ring) for ring in _rings(feature["geometry"]))
        xs = [x for ring in rings for x, _ in ring]
        ys = [y for ring in rings for _, y in ring]
        entries.append(
            (
                str(properties["district_id"]),
                str(properties["canonical_name"]),
                rings,
                (min(xs), min(ys), max(xs), max(ys)),
            )
        )
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


def assign_district(longitude: float, latitude: float) -> tuple[str, str] | None:
    """Locate a lon/lat inside the bundled district boundary, or return ``None``.

    Only the outer ring of each polygon part is tested, so an enclave inside a
    hole would be attributed to the enclosing district.  The bundled boundary is
    a community demo asset, not an authoritative state boundary, so a station
    within a few hundred metres of a district edge may be attributed either way.
    Points outside every polygon return ``None`` rather than a nearest guess.
    """

    for district_id, canonical_name, rings, (west, south, east, north) in _district_rings():
        if not (west <= longitude <= east and south <= latitude <= north):
            continue
        for ring in rings:
            if _point_in_ring(longitude, latitude, list(ring)):
                return district_id, canonical_name
    return None
