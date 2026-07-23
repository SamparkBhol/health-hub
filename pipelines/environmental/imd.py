"""India Meteorological Department live client.

WHAT IS ACTUALLY PUBLIC (probed 2026-07-21, statuses recorded verbatim in
:data:`IMD_ACCESS_EVIDENCE`)
--------------------------------------------------------------------------
The documented gateway at ``api.imd.gov.in/api/v1/*`` is **closed**.  All
seventeen documented endpoints answer ``HTTP 401 {"error":"API key missing"}``
without an account, and ``api.imd.gov.in/public/`` is a registration portal
("Secure JWT", ``register.php`` / ``login.php``).  The older
``mausam.imd.gov.in/api/*.php`` endpoints answer ``HTTP 401`` with the body
``Your IP/Domain <addr> needs to be whitelisted``.  Neither is usable
anonymously, and this module never pretends otherwise: both are surfaced as
typed :class:`~pipelines.environmental.models.ProviderState` values.

What *is* reachable without any credential, and is what this module fetches:

``reactjs.imd.gov.in/geoserver/wfs``
    IMD's own public GeoServer, the backend of the districtwise warning and
    nowcast GIS pages on mausam.imd.gov.in.  OGC WFS 1.1.0, GeoJSON output.
    Layers used here:

    * ``imd:district_warnings_india`` - five-day colour-coded district warning
      grid for all 764 Indian districts, including all 30 in Odisha;
    * ``imd:NowcastWarningDistrict`` - district nowcast with issue/valid times;
    * ``imd:aws_data_layer`` - Automatic Weather Station / ARG observations;
    * ``imd:synop_data_layer`` - SYNOP surface observations with 24-hour
      rainfall;
    * ``imd:metar_data_layer`` - aerodrome observations.

``city.imd.gov.in/citywx/responsive/api/``
    The station directory (``search.php``, GET) and the current city
    observation + seven-day forecast record (``fetchCity_static.php``, POST
    ``ID=<station_id>``).  This carries observed max/min temperature, departure
    from normal, past-24-hour rainfall and relative humidity at 0830/1730 IST.

``cap-sources.s3.amazonaws.com/in-imd-en/``
    IMD's OASIS CAP 1.2 alert feed, XML-signed, published to the Google Public
    Alerts alert-hub.

HONESTY
-------
Everything here is weather.  A warning colour, a nowcast category and a
rainfall total are meteorological facts; none of them is disease surveillance,
and nothing in this module may be rendered as a case count.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import urllib.parse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from workers.ingestion.safe_fetch import FetchError, FetchPolicy, fetch_url

from .districts import assign_district
from .models import AcquisitionState, ProviderState

WFS_HOST = "reactjs.imd.gov.in"
WFS_ENDPOINT = f"https://{WFS_HOST}/geoserver/wfs"
CITY_HOST = "city.imd.gov.in"
CITY_STATION_ENDPOINT = f"https://{CITY_HOST}/citywx/responsive/api/search.php"
CITY_OBSERVATION_ENDPOINT = f"https://{CITY_HOST}/citywx/responsive/api/fetchCity_static.php"
CITY_REFERER = f"https://{CITY_HOST}/citywx/responsive/"
CAP_HOST = "cap-sources.s3.amazonaws.com"
CAP_RSS_URL = f"https://{CAP_HOST}/in-imd-en/rss.xml"

# Odisha's envelope, used to sub-select station layers server-side.
ODISHA_BBOX = (81.3, 17.7, 87.6, 22.7)

MAXIMUM_RESPONSE_BYTES = 20 * 1024 * 1024

#: IMD's own hazard-code table, lifted verbatim from the ``getWarning`` category
#: map embedded in https://mausam.imd.gov.in/responsive/districtWiseWarningGIS.php
HAZARD_CODES: dict[int, str] = {
    1: "No Warning",
    2: "Heavy Rain",
    3: "Heavy Snow",
    4: "Thunderstorms & Lightning, Squall etc",
    5: "Hailstorm",
    6: "Dust Storm",
    7: "Dust Raising Winds",
    8: "Strong Surface Winds",
    9: "Heat Wave",
    10: "Hot Day",
    11: "Warm Night",
    12: "Cold Wave",
    13: "Cold Day",
    14: "Ground Frost",
    15: "Fog",
    16: "Very Heavy Rain",
    17: "Extremely Heavy Rain",
}

#: IMD renders warning severity through a four-step legend on the same page:
#: *No Warning / Watch / Alert / Warning*.  The layer publishes it as an integer
#: ``DayN_Color``.  The mapping below is empirical, not documented: over the
#: 764-district x 5-day national grid retrieved on 2026-07-21, hazard code "1"
#: ("No Warning") carried colour 4 in 1784 of 1786 cells, and the frequency
#: ordering 4 > 3 > 2 > 1 matches a green/yellow/orange/red escalation.  The raw
#: integer is always published next to the label so a consumer can disagree.
WARNING_COLOUR_SEVERITY: dict[int, str] = {
    0: "not_issued",
    1: "warning",
    2: "alert",
    3: "watch",
    4: "no_warning",
}
WARNING_SEVERITY_RANK: dict[str, int] = {
    "not_issued": 0,
    "no_warning": 0,
    "watch": 1,
    "alert": 2,
    "warning": 3,
}
WARNING_COLOUR_BASIS = (
    "empirical: IMD hazard code 1 ('No Warning') carried DayN_Color=4 in 1784 of "
    "1786 national district-days on 2026-07-21; the raw integer is published "
    "alongside every label"
)

#: IMD district spellings that the shared Odisha gazetteer does not carry.
#: Each is an explicit, reviewable judgement, not a fuzzy match.
IMD_DISTRICT_STRING_OVERLAY: dict[str, tuple[str, str]] = {
    "gajapathi": ("OD-DIST-gajapati", "IMD romanisation of Gajapati"),
    "kendraparha": ("OD-DIST-kendrapara", "IMD romanisation of Kendrapara"),
    "nuaparha": ("OD-DIST-nuapada", "IMD romanisation of Nuapada"),
    "rayagarha": ("OD-DIST-rayagada", "IMD romanisation of Rayagada"),
}

#: IMD city-observatory stations that sit inside a known Odisha district.  Only
#: unambiguous stations are listed; an observatory whose district cannot be
#: settled from its name alone is deliberately absent rather than guessed.
CITY_STATION_DISTRICTS: dict[str, tuple[str, str]] = {
    "10002": ("OD-DIST-bhadrak", "Bhadrak (Ranital) observatory, Bhadrak district"),
    "42793": ("OD-DIST-sundargarh", "Rourkela lies in Sundargarh district"),
    "42881": ("OD-DIST-dhenkanal", "Dhenkanal district headquarters observatory"),
    "42882": ("OD-DIST-subarnapur", "Sonepur is the headquarters of Subarnapur district"),
    "42883": ("OD-DIST-sambalpur", "Sambalpur district headquarters observatory"),
    "42886": ("OD-DIST-jharsuguda", "Jharsuguda district headquarters observatory"),
    "42891": ("OD-DIST-keonjhar", "Keonjhargarh is the headquarters of Keonjhar district"),
    "42894": ("OD-DIST-mayurbhanj", "Baripada is the headquarters of Mayurbhanj district"),
    "42895": ("OD-DIST-balasore", "Balasore district headquarters observatory"),
    "42963": ("OD-DIST-balangir", "Bolangir district headquarters observatory"),
    "42964": ("OD-DIST-boudh", "Boudh district headquarters observatory"),
    "42969": ("OD-DIST-angul", "Angul district headquarters observatory"),
    "42971": ("OD-DIST-khordha", "Bhubaneswar airport lies in Khordha district"),
    "42972": ("OD-DIST-nayagarh", "Nayagarh district headquarters observatory"),
    "42973": ("OD-DIST-bhadrak", "Chandbali lies in Bhadrak district"),
    "42976": ("OD-DIST-jagatsinghpur", "Paradip lies in Jagatsinghpur district"),
    "43045": ("OD-DIST-rayagada", "Rayagada district headquarters observatory"),
    "43049": ("OD-DIST-ganjam", "Gopalpur lies in Ganjam district"),
    "43053": ("OD-DIST-puri", "Puri district headquarters observatory"),
    "43091": ("OD-DIST-malkangiri", "Malkangiri district headquarters observatory"),
    "43097": ("OD-DIST-koraput", "Koraput district headquarters observatory"),
    "88833": ("OD-DIST-nuapada", "Nuapada district headquarters observatory"),
    "88834": ("OD-DIST-khordha", "Khordha district headquarters observatory"),
    "93252": ("OD-DIST-jajpur", "Jajpur district headquarters observatory"),
    "93255": ("OD-DIST-deogarh", "Deogarh district headquarters observatory"),
    "93256": ("OD-DIST-bargarh", "Bargarh district headquarters observatory"),
    "93258": ("OD-DIST-nabarangpur", "Nabarangpur district headquarters observatory"),
    "93261": ("OD-DIST-kendrapara", "Kendrapara district headquarters observatory"),
}

#: Probe results for the IMD surfaces that are *not* anonymously usable.  These
#: are recorded so the platform can state precisely what is blocked and why,
#: instead of quietly substituting something else.
IMD_ACCESS_EVIDENCE: tuple[dict[str, str], ...] = (
    {
        "surface": "https://api.imd.gov.in/api/v1/*",
        "probed_at": "2026-07-21",
        "http_status": "401",
        "body": '{"error":"API key missing"}',
        "endpoints_probed": (
            "cityforecast, cityforecast_mapping, current_wx, districtnowcast, "
            "districtrainfall, districtwarning, staterainfall, stationnowcast, "
            "aws_data, aws_data_mapping, state_district_rainfall_forecast, "
            "subdivision_rainfall_forecast, subdivisionwarning, basinqpf, "
            "cyclone_track, cyclone_wind, cyclone_cou"
        ),
        "unlocked_by": (
            "an account on https://api.imd.gov.in/public/register.php; the portal "
            "describes 'Secure JWT' key-based consumption"
        ),
    },
    {
        "surface": "https://mausam.imd.gov.in/api/*.php",
        "probed_at": "2026-07-21",
        "http_status": "401",
        "body": "Your IP/Domain <address> needs to be whitelisted",
        "endpoints_probed": "current_wx_api.php, nowcastapi.php, warnings_district_api.php",
        "unlocked_by": "IMD server-side allowlisting of the caller's source address",
    },
    {
        "surface": "https://dsp.imdpune.gov.in/ (IMD Pune Data Supply Portal)",
        "probed_at": "2026-07-21",
        "http_status": "200",
        "body": "portal reachable; bulk archive supply is an account-and-request service",
        "endpoints_probed": "index.php",
        "unlocked_by": "a Data Supply Portal account and an accepted data request",
    },
)


class IMDValidationError(ValueError):
    """An IMD response did not match the contract this module relies on."""


@dataclass(frozen=True, slots=True)
class IMDReceipt:
    """Provenance for one retrieved IMD product."""

    provider: str
    product: str
    state: AcquisitionState
    requested_url: str
    final_url: str
    retrieved_at: datetime
    sha256: str
    byte_length: int
    feature_count: int
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "product": self.product,
            "state": self.state.value,
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "retrieved_at": self.retrieved_at.isoformat().replace("+00:00", "Z"),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "feature_count": self.feature_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DistrictWarning:
    """IMD's five-day colour-coded warning for one district."""

    district_id: str
    canonical_name: str
    imd_district: str
    issued_for_date: str
    updated_at: str
    days: tuple[dict[str, Any], ...]

    @property
    def peak_severity(self) -> str:
        ranked = max(
            (WARNING_SEVERITY_RANK.get(str(day["severity"]), 0) for day in self.days),
            default=0,
        )
        for label, rank in WARNING_SEVERITY_RANK.items():
            if rank == ranked and label not in {"not_issued"}:
                return label
        return "no_warning"

    def as_dict(self) -> dict[str, Any]:
        return {
            "district_id": self.district_id,
            "canonical_name": self.canonical_name,
            "imd_district": self.imd_district,
            "issued_for_date": self.issued_for_date,
            "updated_at": self.updated_at,
            "peak_severity_next_5_days": self.peak_severity,
            "days": list(self.days),
        }


@dataclass(frozen=True, slots=True)
class StationObservation:
    """One IMD surface observation attributed to a district by containment."""

    network: str
    station: str
    longitude: float
    latitude: float
    district_id: str | None
    canonical_name: str | None
    observed_on: str | None
    observed_at: str | None
    rainfall_mm: float | None
    rainfall_window: str
    relative_humidity_pct: float | None
    temperature_c: float | None
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "network": self.network,
            "station": self.station,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "district_id": self.district_id,
            "canonical_name": self.canonical_name,
            "observed_on": self.observed_on,
            "observed_at": self.observed_at,
            "rainfall_mm": self.rainfall_mm,
            "rainfall_window": self.rainfall_window,
            "relative_humidity_pct": self.relative_humidity_pct,
            "temperature_c": self.temperature_c,
        }


@dataclass(frozen=True, slots=True)
class CityObservation:
    """IMD city-observatory current record plus its seven-day forecast."""

    station_id: str
    station_name: str
    district_id: str
    canonical_name: str
    observed_on: str
    updated_at: str
    rainfall_24h_mm: float | None
    max_temp_c: float | None
    min_temp_c: float | None
    max_departure_from_normal_c: float | None
    humidity_0830_pct: float | None
    humidity_1730_pct: float | None
    forecast: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "district_id": self.district_id,
            "canonical_name": self.canonical_name,
            "observed_on": self.observed_on,
            "updated_at": self.updated_at,
            "rainfall_24h_mm": self.rainfall_24h_mm,
            "max_temp_c": self.max_temp_c,
            "min_temp_c": self.min_temp_c,
            "max_departure_from_normal_c": self.max_departure_from_normal_c,
            "humidity_0830_pct": self.humidity_0830_pct,
            "humidity_1730_pct": self.humidity_1730_pct,
            "forecast": list(self.forecast),
        }


def _normalise(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve_imd_district(name: str, alias_index: dict[str, str]) -> str | None:
    """Map an IMD district spelling to a canonical Odisha district id."""

    key = _normalise(name)
    override = IMD_DISTRICT_STRING_OVERLAY.get(key)
    if override is not None:
        return override[0]
    return alias_index.get(key)


def _number(value: Any) -> float | None:
    """Parse an IMD numeric field, mapping its sentinels to ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "NULL", "NA", "N/A", "-", "99.9", "999", "999.0", "999.00", "-999"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number in {-999.0, 999.0, 99.9} else number


def build_wfs_url(
    typename: str,
    *,
    cql_filter: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    property_names: Sequence[str] | None = None,
    max_features: int | None = None,
) -> str:
    """Compose an OGC WFS 1.1.0 GetFeature request against IMD's GeoServer."""

    if cql_filter is not None and bbox is not None:
        raise ValueError("GeoServer rejects CQL_FILTER and bbox on the same request")
    query: dict[str, str] = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typename": typename,
        "srsname": "EPSG:4326",
        "outputFormat": "application/json",
    }
    if cql_filter is not None:
        query["CQL_FILTER"] = cql_filter
    if bbox is not None:
        query["bbox"] = ",".join(f"{value}" for value in bbox) + ",EPSG:4326"
    if property_names:
        query["propertyName"] = ",".join(property_names)
    if max_features is not None:
        query["maxFeatures"] = str(max_features)
    return f"{WFS_ENDPOINT}?{urllib.parse.urlencode(query)}"


def parse_feature_collection(body: bytes, *, typename: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IMDValidationError(f"{typename} response is not valid UTF-8 JSON") from exc
    if payload.get("type") != "FeatureCollection":
        raise IMDValidationError(f"{typename} response is not a GeoJSON FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise IMDValidationError(f"{typename} response carries no feature list")
    return features


def fetch_wfs(
    typename: str,
    *,
    cql_filter: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    property_names: Sequence[str] | None = None,
    max_features: int | None = None,
    policy: FetchPolicy | None = None,
    product: str | None = None,
) -> tuple[list[dict[str, Any]], IMDReceipt]:
    """Retrieve one IMD GeoServer layer and validate that it is GeoJSON."""

    url = build_wfs_url(
        typename,
        cql_filter=cql_filter,
        bbox=bbox,
        property_names=property_names,
        max_features=max_features,
    )
    result = fetch_url(
        url,
        source_id=f"imd_geoserver_wfs:{typename}",
        allowed_hosts=(WFS_HOST,),
        policy=policy,
        access_path="provider_api",
    )
    features = parse_feature_collection(result.body, typename=typename)
    receipt = IMDReceipt(
        provider="India Meteorological Department",
        product=product or typename,
        state=AcquisitionState.RETRIEVED_AND_VALIDATED,
        requested_url=url,
        final_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        sha256=hashlib.sha256(result.body).hexdigest(),
        byte_length=len(result.body),
        feature_count=len(features),
        warnings=("meteorological_product_not_disease_surveillance",),
    )
    return features, receipt


def fetch_district_warnings(
    *, alias_index: dict[str, str], policy: FetchPolicy | None = None
) -> tuple[tuple[DistrictWarning, ...], IMDReceipt, tuple[str, ...]]:
    """Five-day IMD district warnings for Odisha.

    The layer is national and its ``state`` column is empty, so Odisha rows are
    selected by joining on the ``Obj_id`` values that the nowcast layer - which
    *does* carry a populated ``State`` column - reports for Odisha.
    """

    nowcast_features, _ = fetch_wfs(
        "imd:NowcastWarningDistrict",
        cql_filter="State='ODISHA'",
        policy=policy,
        product="district_nowcast_odisha",
    )
    odisha_ids = {
        feature["properties"].get("Obj_id")
        for feature in nowcast_features
        if feature.get("properties")
    }
    features, receipt = fetch_wfs(
        "imd:district_warnings_india",
        property_names=[
            "District",
            "Date",
            "updated_at",
            *[f"Day_{index}" for index in range(1, 6)],
            *[f"Day{index}_Color" for index in range(1, 6)],
            *[f"Day{index}_text" for index in range(1, 6)],
        ],
        policy=policy,
        product="district_warnings_5day",
    )
    warnings: list[DistrictWarning] = []
    unresolved: list[str] = []
    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("Obj_id") not in odisha_ids:
            continue
        imd_name = str(properties.get("District", "")).strip()
        district_id = resolve_imd_district(imd_name, alias_index)
        if district_id is None:
            unresolved.append(imd_name)
            continue
        days: list[dict[str, Any]] = []
        for index in range(1, 6):
            raw_codes = str(properties.get(f"Day_{index}", "") or "")
            codes = [int(part) for part in re.findall(r"\d+", raw_codes)]
            colour = int(properties.get(f"Day{index}_Color") or 0)
            days.append(
                {
                    "day": index,
                    "hazard_codes": codes,
                    "hazards": [HAZARD_CODES.get(code, f"unmapped_code_{code}") for code in codes],
                    "imd_colour": colour,
                    "severity": WARNING_COLOUR_SEVERITY.get(colour, "unmapped"),
                    "text": str(properties.get(f"Day{index}_text", "") or ""),
                }
            )
        warnings.append(
            DistrictWarning(
                district_id=district_id,
                canonical_name=imd_name.title(),
                imd_district=imd_name,
                issued_for_date=str(properties.get("Date", "")),
                updated_at=str(properties.get("updated_at", "")),
                days=tuple(days),
            )
        )
    warnings.sort(key=lambda item: item.district_id)
    return tuple(warnings), receipt, tuple(sorted(set(unresolved)))


def fetch_district_nowcast(
    *, alias_index: dict[str, str], policy: FetchPolicy | None = None
) -> tuple[tuple[dict[str, Any], ...], IMDReceipt]:
    """IMD district nowcast for Odisha, with its issue and validity times."""

    features, receipt = fetch_wfs(
        "imd:NowcastWarningDistrict",
        cql_filter="State='ODISHA'",
        policy=policy,
        product="district_nowcast_odisha",
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        imd_name = str(properties.get("District", "")).strip()
        district_id = resolve_imd_district(imd_name, alias_index)
        if district_id is None:
            continue
        colour = int(properties.get("Color") or 0)
        categories = [
            index
            for index in range(1, 20)
            if _number(properties.get(f"cat{index}")) not in (None, 0.0)
        ]
        rows.append(
            {
                "district_id": district_id,
                "imd_district": imd_name,
                "date": str(properties.get("Date", "")),
                "issued_at_ist": str(properties.get("toi", "")),
                "valid_upto_ist": str(properties.get("vupto", "")),
                "imd_colour": colour,
                "severity": WARNING_COLOUR_SEVERITY.get(colour, "unmapped"),
                "active_category_slots": categories,
                "message": str(properties.get("message", "") or ""),
                "impact": str(properties.get("impact", "") or ""),
                "action": str(properties.get("action", "") or ""),
                "updated_at": str(properties.get("update_time", "")),
            }
        )
    rows.sort(key=lambda item: str(item["district_id"]))
    return tuple(rows), receipt


# (network, typename, station key, rainfall key, accumulation window, humidity key)
#
# SYNOP and METAR publish an explicitly named ``24hrlyrain`` column, so the
# window is stated.  The AWS/ARG layer publishes a bare ``rainfall`` column and
# IMD's own field reference for ``aws_data`` says only "Fields are
# self-explanatory" - it documents no accumulation period.  Rather than guess
# one, the window is published as undocumented and the observation time is
# carried alongside so a consumer can reason about it.
_STATION_LAYERS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "AWS_ARG",
        "imd:aws_data_layer",
        "station",
        "rainfall",
        "accumulation_window_not_documented_by_provider",
        "rh",
    ),
    ("SYNOP", "imd:synop_data_layer", "station", "24hrlyrain", "past_24_hours", "rh"),
    ("METAR", "imd:metar_data_layer", "station_name", "24hrlyrain", "past_24_hours", "rh"),
)
_TEMPERATURE_KEYS = ("temp", "dbtemp")


def fetch_station_observations(
    *, policy: FetchPolicy | None = None
) -> tuple[tuple[StationObservation, ...], tuple[IMDReceipt, ...]]:
    """AWS/ARG, SYNOP and METAR observations inside the Odisha envelope.

    Every station is attributed to a district by point-in-polygon containment
    against the bundled boundary, so no name matching is involved.  Stations
    inside the bounding box but outside Odisha keep ``district_id=None``.
    """

    observations: list[StationObservation] = []
    receipts: list[IMDReceipt] = []
    for network, typename, name_key, rain_key, rain_window, humidity_key in _STATION_LAYERS:
        features, receipt = fetch_wfs(
            typename,
            bbox=ODISHA_BBOX,
            policy=policy,
            product=f"{network.lower()}_observations_odisha_bbox",
        )
        receipts.append(receipt)
        for feature in features:
            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates")
            if geometry.get("type") != "Point" or not coordinates:
                continue
            longitude, latitude = float(coordinates[0]), float(coordinates[1])
            located = assign_district(longitude, latitude)
            temperature = next(
                (
                    value
                    for key in _TEMPERATURE_KEYS
                    if (value := _number(properties.get(key))) is not None
                ),
                None,
            )
            observations.append(
                StationObservation(
                    network=network,
                    station=str(properties.get(name_key, "") or "").strip(),
                    longitude=longitude,
                    latitude=latitude,
                    district_id=located[0] if located else None,
                    canonical_name=located[1] if located else None,
                    observed_on=str(properties.get("dat", "") or "").rstrip("Z") or None,
                    observed_at=str(
                        properties.get("update_time") or properties.get("utc") or ""
                    ).strip()
                    or None,
                    rainfall_mm=_number(properties.get(rain_key)),
                    rainfall_window=rain_window,
                    relative_humidity_pct=_number(properties.get(humidity_key)),
                    temperature_c=temperature,
                    raw={},
                )
            )
    observations.sort(key=lambda item: (item.network, item.station))
    return tuple(observations), tuple(receipts)


def fetch_city_station_directory(
    *, policy: FetchPolicy | None = None
) -> tuple[dict[str, str], IMDReceipt]:
    """IMD's public city-observatory directory: station id -> station name."""

    result = fetch_url(
        CITY_STATION_ENDPOINT,
        source_id="imd_city_station_directory",
        allowed_hosts=(CITY_HOST,),
        policy=policy,
        access_path="provider_api",
    )
    try:
        payload = json.loads(result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IMDValidationError("city station directory is not valid UTF-8 JSON") from exc
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise IMDValidationError("city station directory carries no data list")
    directory = {
        str(item["station_id"]): str(item["station"]).strip()
        for item in entries
        if isinstance(item, dict) and "station_id" in item and "station" in item
    }
    receipt = IMDReceipt(
        provider="India Meteorological Department",
        product="city_observatory_directory",
        state=AcquisitionState.RETRIEVED_AND_VALIDATED,
        requested_url=CITY_STATION_ENDPOINT,
        final_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        sha256=hashlib.sha256(result.body).hexdigest(),
        byte_length=len(result.body),
        feature_count=len(directory),
    )
    return directory, receipt


def _city_client(policy: FetchPolicy) -> httpx.Client:
    context = ssl.create_default_context()
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=policy.connect_timeout_seconds, read=policy.read_timeout_seconds,
            write=policy.read_timeout_seconds, pool=policy.read_timeout_seconds,
        ),
        follow_redirects=False,
        verify=context,
        headers={
            "User-Agent": policy.user_agent,
            "Referer": CITY_REFERER,
            "Accept": "application/json",
            "Accept-Encoding": policy.accept_encoding,
        },
    )


def fetch_city_observations(
    *,
    stations: dict[str, tuple[str, str]] | None = None,
    policy: FetchPolicy | None = None,
    names: dict[str, str] | None = None,
    station_names: dict[str, str] | None = None,
) -> tuple[tuple[CityObservation, ...], IMDReceipt, tuple[dict[str, str], ...]]:
    """Current city-observatory records for the mapped Odisha stations.

    ``fetchCity_static.php`` answers only to ``POST ID=<station_id>`` with the
    portal's own ``Referer``; a bare ``GET`` is refused with ``HTTP 403``.  The
    shared :func:`fetch_url` helper is GET-only, so this function speaks to the
    single pinned IMD host directly, with the same timeouts, user agent, size
    cap and no-redirect rule as the shared policy.
    """

    selected = stations if stations is not None else CITY_STATION_DISTRICTS
    active_policy = policy or FetchPolicy.load()
    labels = dict(names or {})
    if station_names is None:
        try:
            station_names, _ = fetch_city_station_directory(policy=policy)
        except (FetchError, IMDValidationError):
            # A missing directory costs a friendly station label, nothing more.
            station_names = {}
    observations: list[CityObservation] = []
    failures: list[dict[str, str]] = []
    digest = hashlib.sha256()
    total_bytes = 0
    retrieved_at = datetime.now(UTC)
    with _city_client(active_policy) as client:
        for station_id, (district_id, _basis) in sorted(selected.items()):
            try:
                response = client.post(CITY_OBSERVATION_ENDPOINT, files={"ID": (None, station_id)})
            except httpx.HTTPError as exc:
                failures.append({"station_id": station_id, "code": "transport", "detail": str(exc)})
                continue
            body = response.content[: MAXIMUM_RESPONSE_BYTES + 1]
            if len(body) > MAXIMUM_RESPONSE_BYTES:
                failures.append(
                    {"station_id": station_id, "code": "response_too_large", "detail": "capped"}
                )
                continue
            if response.status_code != 200:
                failures.append(
                    {
                        "station_id": station_id,
                        "code": f"http_{response.status_code}",
                        "detail": body[:120].decode("utf-8", "replace"),
                    }
                )
                continue
            digest.update(body)
            total_bytes += len(body)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                failures.append(
                    {"station_id": station_id, "code": "not_json", "detail": "undecodable body"}
                )
                continue
            if not isinstance(payload, list) or not payload or "dat" not in payload[0]:
                failures.append(
                    {
                        "station_id": station_id,
                        "code": "no_current_record",
                        "detail": str(payload)[:120],
                    }
                )
                continue
            record = payload[0]
            if _number(record.get("max")) is None and _number(record.get("rainfall")) is None:
                failures.append(
                    {
                        "station_id": station_id,
                        "code": "record_all_null",
                        "detail": "station published an empty observation row",
                    }
                )
                continue
            forecast = []
            for day in range(7):
                text = str(record.get(f"forecast{day}", "") or "").strip()
                if not text:
                    continue
                forecast.append(
                    {
                        "day": day + 1,
                        "forecast": text,
                        "max_temp_c": _number(record.get(f"max{day}")),
                        "min_temp_c": _number(record.get(f"min{day}")),
                    }
                )
            observations.append(
                CityObservation(
                    station_id=station_id,
                    station_name=station_names.get(station_id, f"IMD station {station_id}"),
                    district_id=district_id,
                    canonical_name=labels.get(district_id, district_id),
                    observed_on=str(record.get("dat", "")),
                    updated_at=str(record.get("updat", "")),
                    rainfall_24h_mm=_number(record.get("rainfall")),
                    max_temp_c=_number(record.get("max")),
                    min_temp_c=_number(record.get("min")),
                    max_departure_from_normal_c=_number(record.get("maxdep")),
                    humidity_0830_pct=_number(record.get("rh0830")),
                    humidity_1730_pct=_number(record.get("rh1730")),
                    forecast=tuple(forecast),
                )
            )
    receipt = IMDReceipt(
        provider="India Meteorological Department",
        product="city_observatory_current_and_7day_forecast",
        state=(
            AcquisitionState.RETRIEVED_AND_VALIDATED
            if observations
            else AcquisitionState.PROVIDER_UNAVAILABLE
        ),
        requested_url=CITY_OBSERVATION_ENDPOINT,
        final_url=CITY_OBSERVATION_ENDPOINT,
        retrieved_at=retrieved_at,
        sha256=digest.hexdigest(),
        byte_length=total_bytes,
        feature_count=len(observations),
        warnings=(
            "station_point_observation_not_district_average_exposure",
            "meteorological_product_not_disease_surveillance",
        ),
    )
    return tuple(observations), receipt, tuple(failures)


def fetch_cap_alerts(
    *, policy: FetchPolicy | None = None
) -> tuple[tuple[dict[str, Any], ...], IMDReceipt]:
    """IMD's public CAP 1.2 alert index (headline level, from the RSS feed)."""

    result = fetch_url(
        CAP_RSS_URL,
        source_id="imd_cap_alert_feed",
        allowed_hosts=(CAP_HOST,),
        policy=policy,
        access_path="provider_api",
    )
    text = result.body.decode("utf-8", "replace")
    items: list[dict[str, Any]] = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.S):
        entry: dict[str, Any] = {}
        for tag in ("title", "link", "pubDate", "description"):
            match = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S)
            if match:
                entry[tag] = match.group(1).strip()
        if entry:
            items.append(entry)
    receipt = IMDReceipt(
        provider="India Meteorological Department",
        product="cap_1_2_public_alert_feed",
        state=AcquisitionState.RETRIEVED_AND_VALIDATED,
        requested_url=CAP_RSS_URL,
        final_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        sha256=hashlib.sha256(result.body).hexdigest(),
        byte_length=len(result.body),
        feature_count=len(items),
        warnings=(
            "national_feed_areas_are_subdivision_scale_not_district_scale",
            "meteorological_product_not_disease_surveillance",
        ),
    )
    return tuple(items), receipt


def imd_gateway_states(observed_at: datetime | None = None) -> tuple[ProviderState, ...]:
    """Typed states for the IMD surfaces that anonymous access cannot reach."""

    now = observed_at or datetime.now(UTC)
    return tuple(
        ProviderState(
            provider="India Meteorological Department",
            product=evidence["surface"],
            state=(
                AcquisitionState.CREDENTIALS_REQUIRED
                if evidence["http_status"] == "401"
                else AcquisitionState.AWAITING_SOURCE_PERMISSION_OR_APPROVED_API
            ),
            observed_at=now,
            reason=(
                f"probed {evidence['probed_at']}: HTTP {evidence['http_status']} "
                f"{evidence['body']}"
            ),
            metadata=dict(evidence),
        )
        for evidence in IMD_ACCESS_EVIDENCE
    )


def collect_live_imd(
    *,
    alias_index: dict[str, str],
    names: dict[str, str] | None = None,
    policy: FetchPolicy | None = None,
    include_city: bool = True,
) -> dict[str, Any]:
    """Fetch every anonymously reachable IMD product and report what failed."""

    payload: dict[str, Any] = {
        "provider": "India Meteorological Department",
        "collected_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "is_synthetic": False,
        "products": {},
        "receipts": [],
        "failures": [],
        "blocked_surfaces": [
            {
                "provider": state.provider,
                "product": state.product,
                "state": state.state.value,
                "reason": state.reason,
                "unlocked_by": state.metadata.get("unlocked_by"),
            }
            for state in imd_gateway_states()
        ],
        "hazard_code_table": {str(code): label for code, label in HAZARD_CODES.items()},
        "warning_colour_severity": {
            str(code): label for code, label in WARNING_COLOUR_SEVERITY.items()
        },
        "warning_colour_basis": WARNING_COLOUR_BASIS,
    }

    def record(name: str, thunk) -> None:
        try:
            thunk()
        except (FetchError, IMDValidationError, httpx.HTTPError, ValueError) as exc:
            payload["failures"].append(
                {
                    "product": name,
                    "code": getattr(exc, "code", exc.__class__.__name__),
                    "detail": str(exc)[:400],
                }
            )

    def _warnings() -> None:
        rows, receipt, unresolved = fetch_district_warnings(
            alias_index=alias_index, policy=policy
        )
        payload["products"]["district_warnings_5day"] = [item.as_dict() for item in rows]
        payload["receipts"].append(receipt.as_dict())
        if unresolved:
            payload["failures"].append(
                {
                    "product": "district_warnings_5day",
                    "code": "unresolved_district_strings",
                    "detail": ", ".join(unresolved),
                }
            )

    def _nowcast() -> None:
        rows, receipt = fetch_district_nowcast(alias_index=alias_index, policy=policy)
        payload["products"]["district_nowcast"] = list(rows)
        payload["receipts"].append(receipt.as_dict())

    def _stations() -> None:
        rows, receipts = fetch_station_observations(policy=policy)
        payload["products"]["station_observations"] = [item.as_dict() for item in rows]
        payload["receipts"].extend(receipt.as_dict() for receipt in receipts)

    def _city() -> None:
        rows, receipt, failures = fetch_city_observations(policy=policy, names=names)
        payload["products"]["city_observations"] = [item.as_dict() for item in rows]
        payload["receipts"].append(receipt.as_dict())
        payload["failures"].extend(
            {"product": "city_observations", **failure} for failure in failures
        )

    def _cap() -> None:
        rows, receipt = fetch_cap_alerts(policy=policy)
        payload["products"]["cap_alerts"] = list(rows)
        payload["receipts"].append(receipt.as_dict())

    record("district_warnings_5day", _warnings)
    record("district_nowcast", _nowcast)
    record("station_observations", _stations)
    if include_city:
        record("city_observations", _city)
    record("cap_alerts", _cap)
    return payload


def district_signal_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collapse the collected IMD products into one row per district id."""

    index: dict[str, dict[str, Any]] = {}

    def slot(district_id: str) -> dict[str, Any]:
        return index.setdefault(
            district_id,
            {
                "district_id": district_id,
                "imd_warning": None,
                "imd_nowcast": None,
                "imd_city_observation": None,
                "imd_station_observations": [],
            },
        )

    for warning in payload.get("products", {}).get("district_warnings_5day", []):
        slot(str(warning["district_id"]))["imd_warning"] = warning
    for nowcast in payload.get("products", {}).get("district_nowcast", []):
        slot(str(nowcast["district_id"]))["imd_nowcast"] = nowcast
    for city in payload.get("products", {}).get("city_observations", []):
        slot(str(city["district_id"]))["imd_city_observation"] = city
    for station in payload.get("products", {}).get("station_observations", []):
        district_id = station.get("district_id")
        if district_id:
            slot(str(district_id))["imd_station_observations"].append(station)
    return index


def merge_station_rainfall(
    stations: Iterable[dict[str, Any]], *, as_of: str | None = None
) -> dict[str, Any] | None:
    """Summarise a district's station rainfall without inventing a mean.

    The AWS/ARG layer carries stations that stopped reporting months ago but
    still publish their last row.  A stale gauge silently entering a "maximum
    station rainfall" would misrepresent today, so readings are only summarised
    when their observation date equals ``as_of``, and the number excluded is
    published rather than dropped quietly.
    """

    rows = list(stations)
    day = as_of or datetime.now(UTC).date().isoformat()
    fresh = [
        item
        for item in rows
        if item.get("rainfall_mm") is not None and str(item.get("observed_on") or "") == day
    ]
    stale = [
        item
        for item in rows
        if item.get("rainfall_mm") is not None and str(item.get("observed_on") or "") != day
    ]
    if not fresh:
        if not stale:
            return None
        return {
            "as_of": day,
            "station_count": 0,
            "stale_stations_excluded": len(stale),
            "max_station_rainfall_mm": None,
            "median_station_rainfall_mm": None,
            "note": (
                "Every station inside this district last reported before "
                f"{day}, so no current gauge reading is summarised."
            ),
        }
    values = sorted(float(item["rainfall_mm"]) for item in fresh)
    return {
        "as_of": day,
        "station_count": len(values),
        "stale_stations_excluded": len(stale),
        "max_station_rainfall_mm": round(values[-1], 2),
        "median_station_rainfall_mm": round(values[len(values) // 2], 2),
        "note": (
            "Point gauge readings from the stations that fall inside this "
            "district and reported on this date. Not an areal rainfall estimate; "
            "the AWS/ARG accumulation window is not documented by the provider."
        ),
    }
