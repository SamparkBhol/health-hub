"""Public and internal FastAPI surface for the competition profile."""

from __future__ import annotations

import json
import os
import re
import secrets
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from packages.contracts.api import (
    LIVE_EVIDENCE_PLACEHOLDER,
    LIVE_EVIDENCE_REDACTION_STATE,
    AgentQueryRequest,
    CompleteJobRequest,
    DecisionRequest,
    EnqueueJobRequest,
    FailJobRequest,
    JobClaimRequest,
    ReviewClaimRequest,
    SyntheticForecastRequest,
    TranslateRequest,
)
from packages.contracts.envelope import (
    CapabilityState,
    DataVintage,
    DeferralItem,
    Envelope,
    Problem,
    Scoped,
    WarningItem,
    default_context,
    dump_jsonable,
)
from packages.contracts.states import CapabilityCode, LayerType
from packages.forecasting import build_synthetic_report

from .collection_runtime import CollectionRuntime
from .database import (
    Database,
    RepositoryConflict,
    RepositoryNotFound,
    default_database_url,
)
from .evidence_agent import EvidenceAgent

WATERMARK = "SIMULATION_ONLY_NOT_ODISHA_RISK"

GEOGRAPHY_VERSION = "datameet-census-2011-odisha-districts"
DISEASE_VERSION = "odisha-public-health-ontology-v1"


class ApiError(Exception):
    def __init__(self, status: int, code: str, reason_code: str, detail: str) -> None:
        self.status = status
        self.code = code
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(detail)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"req_{uuid.uuid4().hex}")


def _envelope(
    request: Request,
    *,
    layer_type: LayerType,
    data: Any,
    coverage_state: str = "not_applicable",
    warnings: list[WarningItem] | None = None,
    deferrals: list[DeferralItem] | None = None,
    data_vintage: Scoped[DataVintage] | None = None,
    disease_scoped: bool = False,
    geography_scoped: bool = False,
) -> Envelope[Any]:
    return Envelope[Any](
        request_id=_request_id(request),
        context=default_context(
            layer_type,
            coverage_state=coverage_state,  # type: ignore[arg-type]
            data_vintage=data_vintage,
            disease_definition_version=Scoped[str].present(DISEASE_VERSION)
            if disease_scoped
            else Scoped[str].not_applicable(),
            geography_version=Scoped[str].present(GEOGRAPHY_VERSION)
            if geography_scoped
            else Scoped[str].not_applicable(),
        ),
        warnings=warnings or [],
        deferrals=deferrals or [],
        data=data,
    )


def _problem_envelope(request: Request, error: ApiError) -> Envelope[Problem]:
    problem = Problem(
        title=error.code.replace("_", " ").title(),
        status=error.status,
        detail=error.detail,
        code=error.code,
        reason_code=error.reason_code,
        instance=str(request.url.path),
    )
    return Envelope[Problem](
        request_id=_request_id(request),
        context=default_context("not_applicable"),
        warnings=[WarningItem(code=error.code, severity="blocking", message=error.detail)],
        data=problem,
    )


def _idempotency_key(value: str | None) -> str:
    if value is None or not value.strip():
        raise ApiError(
            400,
            "IDEMPOTENCY_KEY_REQUIRED",
            "idempotency_key_required",
            "Idempotency-Key header is required",
        )
    value = value.strip()
    if len(value) > 160 or not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise ApiError(
            400,
            "INVALID_IDEMPOTENCY_KEY",
            "invalid_idempotency_key",
            "Idempotency-Key contains unsupported characters",
        )
    return value


def _check_if_match(if_match: str | None, expected: int) -> None:
    if if_match is None:
        return
    normalised = if_match.strip().strip('"')
    if normalised != str(expected):
        raise ApiError(
            409,
            "STALE_ROW_VERSION",
            "stale_row_version",
            "If-Match does not equal expected_row_version",
        )


def _source_view(row: dict[str, Any]) -> dict[str, Any]:
    if not row["enabled"]:
        state = "policy_pending"
        note = row["policy_state"]
    elif row["last_error_code"]:
        state = "unavailable"
        note = f"Last collection error: {row['last_error_code']}"
    elif row["last_success_at"]:
        state = "ready"
        note = f"Last successful registered retrieval: {row['last_success_at']}"
    else:
        state = "registered_uncontacted"
        note = "Registered and enabled; no successful runtime receipt exists yet."
        if row["source_id"] == "idsp_weekly_outbreaks":
            note += " Live-first/Wayback fallback is configured but not yet observed here."
    return {
        "id": row["source_id"],
        "name": row["name"],
        "language": row["language"].replace(",", " / "),
        "kind": row["content_type"],
        "url": row["canonical_url"],
        "state": state,
        "note": note,
        "lastSuccessAt": row["last_success_at"],
    }


def _signal_view(row: dict[str, Any], source_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = source_by_id.get(row["source_id"], {})
    is_fixture = bool(row.get("is_fixture")) or row["id"].startswith("fixture_")
    district = (row["district_id"] or "district unavailable").removeprefix("OD-DIST-")
    district = district.replace("-", " ").title()
    disease = row["disease"] or "unclassified health issue"
    language = str(row.get("language") or "und")
    review_state = "unreviewed"
    if row.get("review_decision") == "verified":
        review_state = "verified"
    elif row.get("review_decision") in {"rejected", "duplicate"}:
        review_state = "rejected"
    source_name = source.get("name", row["source_id"])
    if is_fixture:
        source_name = f"Bundled synthetic fixture (source-shape: {source_name})"
    return {
        "id": row["id"],
        "title": f"{disease.title()} evidence — {district}",
        "language": language,
        "source": source_name,
        "district": district,
        "disease": disease,
        "assertion": row["assertion"],
        "reviewState": review_state,
        "retrievedAt": row["retrieved_at"],
        "evidence": row["evidence_text"] if is_fixture else LIVE_EVIDENCE_PLACEHOLDER,
        "evidenceVisibility": (
            "synthetic_fixture" if is_fixture else LIVE_EVIDENCE_REDACTION_STATE
        ),
        "hash": row["content_sha256"],
        "snapshotHash": row.get("snapshot_content_sha256"),
        "isFixture": is_fixture,
        "sourceSnapshotId": row["source_snapshot_id"],
        "accessPath": row.get("access_path"),
        "canonicalUrl": row.get("registered_source_url"),
        "canonicalUrlState": "registered_source_only_detail_url_withheld",
        "processingState": row.get("processing_state"),
        "redactionState": row.get("redaction_state"),
        "eventReviewEligible": bool(row.get("event_review_eligible")),
        "extractorVersion": row.get("extractor_version"),
    }


def _load_boundary_asset() -> tuple[dict[str, Any], dict[str, Any], str]:
    project_root = Path(__file__).resolve().parents[2]
    asset_path = project_root / "data" / "boundaries" / "odisha_districts_census_2011.geojson"
    manifest_path = project_root / "data" / "boundaries" / "manifest.json"
    if not asset_path.is_file() or not manifest_path.is_file():
        raise ApiError(
            503,
            "BOUNDARY_ASSET_UNAVAILABLE",
            "boundary_asset_unavailable",
            "The pinned district boundary or its manifest is unavailable.",
        )
    raw = asset_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256(raw).hexdigest()
    if digest != manifest.get("asset_sha256"):
        raise ApiError(
            503,
            "BOUNDARY_CHECKSUM_MISMATCH",
            "boundary_checksum_mismatch",
            "The district boundary does not match its pinned manifest.",
        )
    geojson = json.loads(raw)
    if geojson.get("type") != "FeatureCollection" or len(geojson.get("features", [])) != 30:
        raise ApiError(
            503,
            "BOUNDARY_VALIDATION_FAILED",
            "boundary_validation_failed",
            "The competition boundary must contain exactly 30 Odisha district features.",
        )
    return geojson, manifest, digest


_REAL_FORECAST_GROUPS = frozenset(
    {"any_reported_outbreak", "diarrhoeal_and_cholera", "vector_borne"}
)
_REAL_FORECAST_HORIZONS = frozenset({1, 2, 4, 8, 12})


def _not_incidence_warning() -> WarningItem:
    """The blocking caption every real-model response must carry."""

    from packages.forecasting import NOT_INCIDENCE_WARNING

    return WarningItem(code="NOT_INCIDENCE", severity="blocking", message=NOT_INCIDENCE_WARNING)


@contextmanager
def _forecast_artefact_guard() -> Iterator[None]:
    """Translate a missing or corrupt model artefact into a typed 503.

    The artefact is committed, so this should not fire in a healthy deployment; if
    it does, the service must say the model is unavailable rather than degrade into
    an unlabelled or improvised number.
    """

    from packages.forecasting import ForecastArtefactInvalid, ForecastArtefactMissing

    try:
        yield
    except (ForecastArtefactMissing, ForecastArtefactInvalid) as exc:
        raise ApiError(
            503,
            "REAL_FORECAST_ARTEFACT_UNAVAILABLE",
            "forecast_artefact_unavailable",
            str(exc),
        ) from exc


@lru_cache(maxsize=1)
def _district_universe() -> tuple[dict[str, str], ...]:
    """Return the canonical 30-district universe from the pinned boundary asset.

    The signal map deliberately omits districts that have no evidence, because a
    missing row means "unknown", never "zero disease". A client still has to draw
    all thirty districts, so it needs the full universe separately rather than
    inferring absence from an aggregate it cannot see.
    """

    geojson, _manifest, _digest = _load_boundary_asset()
    universe: list[dict[str, str]] = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        district_id = properties.get("district_id")
        if not district_id:
            continue
        universe.append(
            {
                "district_id": district_id,
                "canonical_name": properties.get("canonical_name", district_id),
            }
        )
    return tuple(sorted(universe, key=lambda item: item["canonical_name"]))


def _districts_with_observation_state(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the 30-district universe with an explicit typed observation state.

    `observed` carries the aggregate; `unknown` carries no count at all. A count of
    zero is never emitted, because the platform cannot distinguish "no evidence was
    published" from "no disease occurred".
    """

    observed = {row["district_id"]: row for row in rows}
    projected: list[dict[str, Any]] = []
    for district in _district_universe():
        district_id = district["district_id"]
        row = observed.get(district_id)
        if row is None:
            projected.append(
                {
                    **district,
                    "observation_state": "unknown_no_published_evidence_retrieved",
                }
            )
        else:
            projected.append({**district, "observation_state": "observed", **row})
    return projected


def _load_epiclim_audit() -> dict[str, Any]:
    audit_path = Path(__file__).resolve().parents[2] / "data" / "epiclim" / "audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            503,
            "EPICLIM_AUDIT_UNAVAILABLE",
            "epiclim_audit_unavailable",
            f"The frozen EpiClim audit cannot be loaded: {exc}",
        ) from exc
    valid = (
        audit.get("schema_version") == "1.0.0"
        and audit.get("national", {}).get("rows") == 8985
        and audit.get("odisha", {}).get("rows") == 358
        and audit.get("eligibility", {}).get("district_week_count_forecast") == "ineligible"
    )
    if not valid:
        raise ApiError(
            503,
            "EPICLIM_AUDIT_VALIDATION_FAILED",
            "epiclim_audit_validation_failed",
            "The frozen EpiClim audit failed its invariant checks.",
        )
    return audit


def _signal_coverage_state(rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    if rows and all(bool(row.get("is_fixture")) for row in rows):
        return "fixture_fallback"
    enabled_sources = [source for source in sources if bool(source.get("enabled"))]
    successful_sources = [source for source in enabled_sources if source.get("last_success_at")]
    if enabled_sources and len(successful_sources) == len(enabled_sources):
        return "observed_for_registered_sources"
    if successful_sources:
        return "partial_registered_source_contact"
    return "unknown"


def _public_signal_fixture_mode(database: Database) -> str:
    """Never blend fixtures with live evidence after a live signal exists.

    An index receipt alone is not evidence and must not blank the public demo.
    """

    return "live_only" if database.has_live_signals() else "fixture_only"


def _normalise_rfc3339_filter(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(
            422,
            "INVALID_TIME_FILTER",
            f"{field}_must_be_rfc3339",
            f"{field} must be an RFC3339 timestamp with an explicit UTC offset.",
        ) from exc
    if parsed.tzinfo is None:
        raise ApiError(
            422,
            "INVALID_TIME_FILTER",
            f"{field}_timezone_required",
            f"{field} must include Z or an explicit UTC offset.",
        )
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def create_app(database_url: str | None = None) -> FastAPI:
    explicitly_injected_database = database_url is not None
    application = FastAPI(
        title="Janaswasthya Agentic Public Health Intelligence API",
        version="1.0.0",
        description=(
            "Multilingual public-health intelligence for Odisha: crawls registered "
            "Odia, Hindi and English sources; preserves redacted evidence and "
            "provenance; serves district maps and environmental outlooks; and routes "
            "candidate alerts to human verification. Historical catalogue experiments "
            "are kept separate from operational disease claims."
        ),
    )
    database = Database(database_url or default_database_url())
    application.state.database = database
    collection_runtime = CollectionRuntime(database)
    application.state.collection_runtime = collection_runtime
    evidence_agent = EvidenceAgent.create(database)
    application.state.evidence_agent = evidence_agent
    if os.getenv("AUTO_REPLAY_FIXTURES", "false").casefold() == "true":
        application.router.add_event_handler("startup", database.replay_demo_fixtures)
    application.router.add_event_handler("startup", collection_runtime.start)
    application.router.add_event_handler("shutdown", collection_runtime.stop)
    allowed_origins = [
        item.strip()
        for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if item.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Content-Type",
            "Idempotency-Key",
            "If-Match",
            "X-Collector-Token",
            "X-Demo-Token",
            "X-Operator-ID",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Boundary-Authority",
            "X-Boundary-Vintage",
            "X-Boundary-SHA256",
            "X-Attribution",
        ],
        max_age=600,
    )

    @application.middleware("http")
    async def correlation_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", supplied)
            else f"req_{uuid.uuid4().hex}"
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        # JSON APIs should not rely on browser MIME sniffing or be embedded in
        # another origin.  Public immutable assets may set their own cache
        # policy in the route; everything else is non-cacheable by default.
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content=dump_jsonable(_problem_envelope(request, error)),
            headers={"X-Request-ID": _request_id(request)},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        wrapped = ApiError(
            422,
            "REQUEST_VALIDATION_FAILED",
            "request_validation_failed",
            "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            ),
        )
        return JSONResponse(
            status_code=422,
            content=dump_jsonable(_problem_envelope(request, wrapped)),
            headers={"X-Request-ID": _request_id(request)},
        )

    def require_demo_token(x_demo_token: str | None = Header(default=None)) -> None:
        writes_enabled = os.getenv("PUBLIC_WRITE_ENABLED", "false").casefold() == "true"
        if not explicitly_injected_database and not writes_enabled:
            raise ApiError(
                503,
                "PUBLIC_WRITES_DISABLED",
                "public_writes_disabled",
                "Public demo mutations are disabled for this deployment.",
            )
        expected = os.getenv("DEMO_API_TOKEN")
        if expected and (
            x_demo_token is None or not secrets.compare_digest(x_demo_token, expected)
        ):
            raise ApiError(
                401, "DEMO_AUTH_REQUIRED", "demo_auth_required", "valid X-Demo-Token is required"
            )
        if not expected and not explicitly_injected_database:
            raise ApiError(
                503,
                "DEMO_MUTATIONS_DISABLED",
                "demo_token_not_configured",
                "Demo mutations are disabled until DEMO_API_TOKEN is configured.",
            )

    def require_collector_token(x_collector_token: str | None = Header(default=None)) -> None:
        expected = os.getenv("COLLECTOR_API_TOKEN")
        if expected and (
            x_collector_token is None or not secrets.compare_digest(x_collector_token, expected)
        ):
            raise ApiError(
                401,
                "COLLECTOR_AUTH_REQUIRED",
                "collector_auth_required",
                "valid X-Collector-Token is required",
            )
        if not expected and not explicitly_injected_database:
            raise ApiError(
                503,
                "COLLECTOR_DISABLED",
                "collector_token_not_configured",
                "Collector mutations are disabled until COLLECTOR_API_TOKEN is configured.",
            )

    def translate_repository_error(error: Exception) -> ApiError:
        if isinstance(error, RepositoryNotFound):
            return ApiError(404, "RESOURCE_NOT_FOUND", "resource_not_found", str(error))
        if isinstance(error, RepositoryConflict):
            return ApiError(409, error.code, error.code.lower(), error.detail)
        raise error

    @application.get("/", response_model=None)
    def service_root() -> RedirectResponse | dict[str, Any]:
        """Open the web application when configured, otherwise describe the API."""

        web_url = os.getenv("WEB_APP_URL", "").strip()
        if web_url:
            return RedirectResponse(url=web_url, status_code=307)
        return {
            "name": "Janaswasthya Agentic Public Health Intelligence API",
            "status": "alive",
            "web_url": None,
            "documentation": "/docs",
            "health": "/api/v1/healthz",
        }

    @application.get("/api/v1/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "alive", "scope": "process_liveness_only"}

    @application.get("/api/v1/readyz")
    def readyz(request: Request, response: Response) -> Envelope[dict[str, Any]]:
        database_ready = database.ready()
        boundary_ready = True
        try:
            _load_boundary_asset()
        except ApiError:
            boundary_ready = False
        ready = database_ready and boundary_ready
        if not ready:
            response.status_code = 503
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial" if ready else "unavailable",
            geography_scoped=True,
            data={
                "ready": ready,
                "database": "ready" if database_ready else "unavailable",
                "database_backend": database.backend,
                "boundary": "community_demo_boundary" if boundary_ready else "unavailable",
                "snapshot": "runtime_database",
            },
        )

    @application.get("/api/v1/readiness")
    def readiness(request: Request) -> Envelope[dict[str, Any]]:
        capabilities = [
            {
                "capability": "registered_source_evidence",
                "state": {"code": "available"},
                "operational": True,
            },
            {
                "capability": "district_geometry",
                "state": {"code": "community_demo_boundary"},
                "operational": True,
            },
            {
                "capability": "cross_language_dedup",
                "state": {"code": "dedup_cross_language_unvalidated"},
                "operational": False,
            },
            {
                "capability": "official_public_disease_maps",
                "state": {"code": "available"},
                "operational": True,
            },
            {
                "capability": "authorised_routine_surveillance",
                "state": {"code": "awaiting_sponsor_data"},
                "operational": False,
            },
            {
                "capability": "public_three_month_research_outlook",
                "state": {"code": "research_only_not_operational_alert"},
                "operational": True,
            },
            {
                "capability": "authorised_operational_outbreak_forecast",
                "state": {"code": "target_series_ineligible"},
                "operational": False,
            },
            {
                "capability": "tahasil_health_map",
                "state": {"code": "not_implemented_phase_one"},
                "operational": False,
                # Not a backlog item. Odisha's revenue geography and its health
                # reporting geography are different tessellations, so a
                # tahasil-level health map would be a join that does not exist.
                "detail": (
                    "Odisha has 317 revenue tahasils but 314 community development "
                    "blocks, and public-health reporting runs through blocks and "
                    "CHC/PHC catchments rather than tahasils. The two boundary sets "
                    "are not nested, so allocating block or facility-catchment "
                    "reports to tahasils would be a spatial fiction rather than a "
                    "finer measurement. No authorised sub-district surveillance "
                    "series is available to build or validate such a map, so the "
                    "map stops at the district, which is the finest unit the "
                    "bundled official series actually reports."
                ),
            },
        ]
        # Validate all codes against the closed union before returning them.
        for capability in capabilities:
            CapabilityState.model_validate(capability["state"])
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial",
            geography_scoped=True,
            data={"profile": "production_shaped", "capabilities": capabilities},
        )

    @application.get("/api/v1/collector/status")
    def collector_status(request: Request) -> Envelope[dict[str, Any]]:
        status = collection_runtime.status()
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial",
            data=status,
            warnings=(
                []
                if status["contact_configured"]
                else [
                    WarningItem(
                        code="CRAWLER_CONTACT_NOT_CONFIGURED",
                        severity="blocking",
                        message=(
                            "Live scheduled crawling remains withheld until a monitored "
                            "crawler contact is configured."
                        ),
                    )
                ]
            ),
        )

    @application.post("/api/v1/internal/collector/tick")
    def collector_tick(
        request: Request,
        maximum_jobs: int = Query(default=1, ge=1, le=1),
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        result = collection_runtime.tick(maximum_jobs=maximum_jobs)
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial",
            data=result,
            warnings=(
                [
                    WarningItem(
                        code=str(result["reason_code"]),
                        severity="blocking",
                        message="The collector was safely withheld; no live request was made.",
                    )
                ]
                if result["state"] == "withheld"
                else []
            ),
        )

    @application.get("/api/v1/internal/collector/pending-pdfs")
    def pending_pdf_approvals(
        request: Request,
        include_inspection_url: bool = Query(default=False),
        x_operator_id: str | None = Header(default=None, alias="X-Operator-ID"),
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        """List byte digests awaiting deliberate operator promotion.

        This never returns the discovered URL, anchor label, response body or
        extracted text.  Adding a digest to APPROVED_PDF_SHA256S authorises a
        new bounded parsing attempt; it is not a malware or rights verdict.
        """

        if include_inspection_url and (
            x_operator_id is None or not re.fullmatch(r"[A-Za-z0-9._:@-]{3,100}", x_operator_id)
        ):
            raise ApiError(
                400,
                "OPERATOR_ID_REQUIRED",
                "operator_id_required_for_sensitive_read",
                "A valid X-Operator-ID is required to reveal protected inspection URLs.",
            )
        items = database.list_pending_pdf_approvals(include_inspection_url=include_inspection_url)
        if include_inspection_url:
            assert x_operator_id is not None
            database.record_sensitive_operator_read(
                operator_id=x_operator_id,
                resource="pending_pdf_inspection_urls",
                item_count=len(items),
            )
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial",
            data={
                "count": len(items),
                "approval_environment_variable": "APPROVED_PDF_SHA256S",
                "items": items,
            },
            warnings=[
                WarningItem(
                    code="DIGEST_APPROVAL_IS_NOT_A_SAFETY_VERDICT",
                    severity="warning",
                    message=(
                        "Promotion authorises bounded parsing of those exact bytes; "
                        "it does not establish malware safety, copyright permission "
                        "or document accuracy."
                    ),
                )
            ],
        )

    @application.get("/api/v1/sources")
    def sources(request: Request) -> Envelope[list[dict[str, Any]]]:
        return _envelope(
            request,
            layer_type="coverage",
            coverage_state="partial",
            data=[_source_view(row) for row in database.list_sources()],
            warnings=[
                WarningItem(
                    code="REGISTERED_SOURCES_ONLY",
                    severity="info",
                    message="Coverage applies only to the listed sources and collection receipts.",
                )
            ],
        )

    @application.get("/api/v1/boundaries/districts")
    def district_boundaries(request: Request) -> JSONResponse:
        """Serve the pinned community-demo boundary as native GeoJSON.

        Geometry is kept outside the response envelope so MapLibre can consume
        it directly.  Epistemic status, attribution, vintage and checksum are
        carried both in the feature properties and response headers.
        """

        geojson, manifest, digest = _load_boundary_asset()
        return JSONResponse(
            content=geojson,
            media_type="application/geo+json",
            headers={
                "X-Request-ID": _request_id(request),
                "X-Boundary-Authority": str(manifest["geometry_authority"]),
                "X-Boundary-Vintage": str(manifest["source_vintage"]),
                "X-Boundary-SHA256": digest,
                "X-Attribution": str(manifest["attribution"]).replace("—", "-"),
                "Link": '<https://creativecommons.org/licenses/by/2.5/in/>; rel="license"',
                "Cache-Control": "public, max-age=86400, immutable",
            },
        )

    @application.get("/api/v1/signals")
    def signals(
        request: Request,
        district_id: str | None = Query(default=None, max_length=100),
        disease: str | None = Query(default=None, max_length=100),
        language: Literal["or", "hi", "en", "mixed", "und"] | None = Query(default=None),
        assertion: Literal["affirmed", "not_affirmed", "speculative", "non_current", "all"] = Query(
            default="affirmed"
        ),
        retrieved_from: str | None = Query(default=None, max_length=40),
        retrieved_to: str | None = Query(default=None, max_length=40),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> Envelope[list[dict[str, Any]]]:
        source_rows = database.list_sources()
        source_by_id = {row["source_id"]: row for row in source_rows}
        fixture_mode = _public_signal_fixture_mode(database)
        normalised_from = _normalise_rfc3339_filter(retrieved_from, "retrieved_from")
        normalised_to = _normalise_rfc3339_filter(retrieved_to, "retrieved_to")
        if normalised_from and normalised_to and normalised_from > normalised_to:
            raise ApiError(
                422,
                "INVALID_TIME_RANGE",
                "retrieved_from_after_retrieved_to",
                "retrieved_from must be earlier than or equal to retrieved_to.",
            )
        rows = database.list_signals(
            district_id=district_id,
            disease=disease,
            language=language,
            assertion=None if assertion == "all" else assertion,
            retrieved_from=normalised_from,
            retrieved_to=normalised_to,
            fixture_mode=fixture_mode,
            public_only=True,
            limit=limit,
        )
        coverage_state = (
            "fixture_fallback"
            if fixture_mode == "fixture_only" and rows
            else _signal_coverage_state(rows, source_rows)
        )
        warnings = [
            WarningItem(
                code="NOT_INCIDENCE",
                severity="warning",
                message="Published evidence density is not disease incidence or burden.",
            )
        ]
        if coverage_state == "unknown":
            warnings.append(
                WarningItem(
                    code="NO_SUCCESSFUL_COLLECTION_RECEIPT",
                    severity="blocking",
                    message=(
                        "No successful live collection receipt exists; empty cells are "
                        "unknown, not zero published items."
                    ),
                )
            )
        return _envelope(
            request,
            layer_type="public_source_signal",
            coverage_state=coverage_state,
            geography_scoped=True,
            disease_scoped=True,
            data=[_signal_view(row, source_by_id) for row in rows],
            warnings=warnings,
        )

    @application.get("/api/v1/maps/published-signals")
    def published_signal_map(
        request: Request,
        disease: str | None = Query(default=None, max_length=100),
        language: Literal["or", "hi", "en", "mixed", "und"] | None = Query(default=None),
        assertion: Literal["affirmed", "not_affirmed", "speculative", "non_current", "all"] = Query(
            default="affirmed"
        ),
        retrieved_from: str | None = Query(default=None, max_length=40),
        retrieved_to: str | None = Query(default=None, max_length=40),
    ) -> Envelope[dict[str, Any]]:
        """Aggregate unverified published mentions without manufacturing zeroes."""

        source_rows = database.list_sources()
        fixture_mode = _public_signal_fixture_mode(database)
        normalised_from = _normalise_rfc3339_filter(retrieved_from, "retrieved_from")
        normalised_to = _normalise_rfc3339_filter(retrieved_to, "retrieved_to")
        if normalised_from and normalised_to and normalised_from > normalised_to:
            raise ApiError(
                422,
                "INVALID_TIME_RANGE",
                "retrieved_from_after_retrieved_to",
                "retrieved_from must be earlier than or equal to retrieved_to.",
            )
        rows = database.aggregate_signal_counts(
            disease=disease,
            language=language,
            assertion=None if assertion == "all" else assertion,
            retrieved_from=normalised_from,
            retrieved_to=normalised_to,
            fixture_mode=fixture_mode,
        )
        coverage_state = (
            "fixture_fallback"
            if fixture_mode == "fixture_only" and rows
            else _signal_coverage_state(rows, source_rows)
        )
        return _envelope(
            request,
            layer_type="public_source_signal",
            coverage_state=coverage_state,
            geography_scoped=True,
            disease_scoped=True,
            data={
                "metric": "published_signal_count",
                "time_axis": "retrieval_time_not_event_onset",
                "fixture_mode": fixture_mode,
                "filters": {
                    "disease": disease,
                    "language": language,
                    "assertion": assertion,
                    "retrieved_from": normalised_from,
                    "retrieved_to": normalised_to,
                },
                "districts": rows,
                # The full 30-district universe with a typed observation state, so a
                # client can draw every district without inventing a zero for the
                # ones that simply have no retrieved evidence.
                "district_universe": _districts_with_observation_state(rows),
                "district_universe_size": len(_district_universe()),
            },
            warnings=[
                WarningItem(
                    code="NOT_DISEASE_INCIDENCE",
                    severity="blocking",
                    message=(
                        "Values are counts of retrieved published evidence records, not "
                        "cases, prevalence, burden, event dates or unique outbreaks."
                    ),
                ),
                WarningItem(
                    code="MISSING_IS_UNKNOWN",
                    severity="warning",
                    message="Districts without a row are unknown, never zero disease.",
                ),
            ],
        )

    @application.post("/api/v1/agent/query")
    def agent_query(
        payload: AgentQueryRequest,
        request: Request,
    ) -> Envelope[dict[str, Any]]:
        result = evidence_agent.answer(
            payload.question,
            district_id=payload.district_id,
            disease=payload.disease,
            maximum_evidence=payload.maximum_evidence,
            target_language=payload.target_language,
            history=[turn.model_dump() for turn in payload.history],
        )
        warnings = [
            WarningItem(
                code="EVIDENCE_ASSISTANT_NOT_CLINICAL_ADVICE",
                severity="info",
                message=(
                    "This source-grounded agent retrieves platform evidence, generates a "
                    "cited answer and supports human-reviewed candidate alerts; it does not "
                    "provide diagnosis or treatment."
                ),
            )
        ]
        if result["answer_state"] in {
            "insufficient_training_data",
            "not_observable_from_public_sources",
            "ambiguous_scope",
            "out_of_scope_clinical",
        }:
            warnings.append(
                WarningItem(
                    code=str(result["reason_codes"][0]),
                    severity="blocking",
                    message="The requested claim is unavailable and no number was generated.",
                )
            )
        source_rows = database.list_sources()
        evidence_rows = result.get("evidence", [])
        evidence_coverage = _signal_coverage_state(evidence_rows, source_rows)
        return _envelope(
            request,
            layer_type=(
                "forecast" if result["intent"] == "forecast_request" else "public_source_signal"
            ),
            coverage_state=(
                "awaiting_sponsor_data"
                if result["intent"] in {"forecast_request", "incidence_request"}
                else evidence_coverage
            ),
            geography_scoped=True,
            disease_scoped=True,
            data=result,
            warnings=warnings,
        )

    @application.get("/api/v1/review/tasks")
    def review_tasks(
        request: Request,
        limit: int = Query(default=100, ge=1, le=200),
        _: None = Depends(require_demo_token),
    ) -> Envelope[list[dict[str, Any]]]:
        return _envelope(
            request,
            layer_type="not_applicable",
            data=database.list_review_tasks(limit=limit),
        )

    @application.post("/api/v1/review/tasks/{task_id}/claim")
    def claim_review_task(
        task_id: str,
        payload: ReviewClaimRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        _: None = Depends(require_demo_token),
    ) -> Envelope[dict[str, Any]]:
        key = _idempotency_key(idempotency_key)
        _check_if_match(if_match, payload.expected_row_version)
        try:
            task, replayed = database.claim_review_task(
                task_id=task_id,
                reviewer_id=payload.reviewer_id,
                expected_row_version=payload.expected_row_version,
                lease_seconds=payload.lease_seconds,
                idempotency_key=key,
            )
        except (RepositoryConflict, RepositoryNotFound) as error:
            raise translate_repository_error(error) from error
        return _envelope(
            request,
            layer_type="not_applicable",
            data={"task": task, "idempotent_replay": replayed},
        )

    @application.post("/api/v1/review/tasks/{task_id}/decision")
    def decide_review_task(
        task_id: str,
        payload: DecisionRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        _: None = Depends(require_demo_token),
    ) -> Envelope[dict[str, Any]]:
        key = _idempotency_key(idempotency_key)
        _check_if_match(if_match, payload.expected_row_version)
        try:
            decision, replayed = database.decide_review_task(
                task_id=task_id,
                reviewer_id=payload.reviewer_id,
                expected_row_version=payload.expected_row_version,
                decision=payload.decision,
                rationale=payload.rationale,
                supersedes_decision_id=payload.supersedes_decision_id,
                event=payload.event.model_dump() if payload.event else None,
                idempotency_key=key,
            )
        except (RepositoryConflict, RepositoryNotFound) as error:
            raise translate_repository_error(error) from error
        return _envelope(
            request,
            layer_type="not_applicable",
            data={"decision": decision, "idempotent_replay": replayed},
        )

    @application.get("/api/v1/layers/{layer_type}")
    def layer(
        layer_type: Literal[
            "public_source_signal",
            "verified_event",
            "official_event_catalogue",
            "observed_surveillance",
            "forecast",
            "coverage",
        ],
        request: Request,
        limit: int = Query(default=100, ge=1, le=200),
        disease: str | None = Query(default=None, max_length=100),
    ) -> Envelope[Any]:
        if layer_type == "public_source_signal":
            source_rows = database.list_sources()
            source_by_id = {row["source_id"]: row for row in source_rows}
            fixture_mode = _public_signal_fixture_mode(database)
            rows = database.list_signals(
                limit=limit,
                fixture_mode=fixture_mode,
                assertion="affirmed",
                public_only=True,
            )
            return _envelope(
                request,
                layer_type=layer_type,
                data=[_signal_view(row, source_by_id) for row in rows],
                coverage_state=_signal_coverage_state(rows, source_rows),
                disease_scoped=True,
                geography_scoped=True,
                warnings=[
                    WarningItem(
                        code="NOT_INCIDENCE",
                        severity="warning",
                        message="Article and signal counts are not cases.",
                    )
                ],
            )
        if layer_type == "verified_event":
            return _envelope(
                request,
                layer_type=layer_type,
                data=database.list_verified_events(limit=limit),
                coverage_state="partial",
                disease_scoped=True,
                geography_scoped=True,
            )
        if layer_type == "official_event_catalogue":
            return _envelope(
                request,
                layer_type=layer_type,
                data=database.list_catalogue_events(limit=limit),
                coverage_state="partial",
                disease_scoped=True,
                geography_scoped=True,
                deferrals=[
                    DeferralItem(
                        capability="routine_case_counts",
                        state=CapabilityState(code="public_catalogue_only_no_denominator"),
                        reason_code="POSITIVE_ONLY_CATALOGUE",
                    )
                ],
            )
        if layer_type == "observed_surveillance":
            from packages.forecasting.authorised_surveillance import SurveillanceContractError
            from packages.forecasting.operational import observed_surveillance_map

            try:
                observed = observed_surveillance_map(disease=disease)
            except SurveillanceContractError:
                observed = None
            if observed is not None:
                records = observed["records"][:limit]
                completeness_warning = WarningItem(
                    code="OBSERVED_SURVEILLANCE_RATE",
                    severity="info",
                    message=(
                        "Map values are official observed weekly cases per 100,000, not "
                        "model output. Read case-volume completeness with every district."
                    ),
                )
                return _envelope(
                    request,
                    layer_type=layer_type,
                    data=records,
                    coverage_state=(
                        "observed_for_registered_sources"
                        if all(item["observation_state"] == "observed_complete" for item in records)
                        else "partial"
                    ),
                    disease_scoped=True,
                    geography_scoped=True,
                    warnings=[completeness_warning],
                )
            return _envelope(
                request,
                layer_type=layer_type,
                data=[],
                coverage_state="awaiting_sponsor_data",
                disease_scoped=True,
                geography_scoped=True,
                deferrals=[
                    DeferralItem(
                        capability="observed_surveillance",
                        state=CapabilityState(code="awaiting_sponsor_data"),
                        reason_code="AUTHORISED_ROUTINE_AGGREGATES_NOT_SUPPLIED",
                    )
                ],
            )
        if layer_type == "forecast":
            return _real_forecast_read(request)
        return _envelope(
            request,
            layer_type="coverage",
            data=database.list_sources(),
            coverage_state="partial",
        )

    def _real_forecast_read(request: Request) -> Envelope[list[Any]]:
        return _envelope(
            request,
            layer_type="forecast",
            data=[],
            coverage_state="awaiting_sponsor_data",
            geography_scoped=True,
            data_vintage=Scoped[DataVintage].unavailable("NO_ELIGIBLE_ODISHA_TARGET_SERIES"),
            warnings=[
                WarningItem(
                    code="NOT_A_FORECAST",
                    severity="blocking",
                    message="No authorised Odisha outbreak probability is available.",
                )
            ],
            deferrals=[
                DeferralItem(
                    capability="odisha_forecast",
                    state=CapabilityState(code="awaiting_sponsor_data"),
                    reason_code="IHIP_VINTAGES_NOT_PROVEN",
                )
            ],
        )

    @application.get("/api/v1/forecast")
    def forecast_read(request: Request) -> Envelope[list[Any]]:
        return _real_forecast_read(request)

    @application.get("/api/v1/forecast/operational/readiness")
    def operational_forecast_readiness(request: Request) -> Envelope[dict[str, Any]]:
        """State the exact aggregate-data gate for a real outbreak model.

        This route is intentionally useful before data arrive: it gives the State
        Surveillance Unit a no-PII CSV contract and reports why an export is or
        is not fit for training. It never treats a missing file as zero disease.
        """

        from packages.forecasting.authorised_surveillance import audit_export

        data = audit_export()
        eligible = bool(data.get("eligible_for_training"))
        status = str(data.get("status"))
        warning = WarningItem(
            code=(
                "OPERATIONAL_FORECAST_DATA_READY" if eligible else "OPERATIONAL_FORECAST_DATA_GATE"
            ),
            severity="info" if eligible else "warning",
            message=(
                "Authorised aggregate surveillance data satisfy the initial structural "
                "training gate. A disease-specific backtest and calibration gate remains "
                "mandatory before publishing."
                if eligible
                else "No operational outbreak probability is issued until the authorised "
                "aggregate export passes the contract, completeness and history checks."
            ),
        )
        deferrals = (
            []
            if eligible
            else [
                DeferralItem(
                    capability="authorised_district_week_surveillance_forecast",
                    state=CapabilityState(
                        code=(
                            "awaiting_sponsor_data"
                            if status == "awaiting_authorised_aggregate_export"
                            else "insufficient_evidence"
                        )
                    ),
                    reason_code=(data.get("reason_codes") or ["SURVEILLANCE_EXPORT_UNAVAILABLE"])[
                        0
                    ],
                )
            ]
        )
        return _envelope(
            request,
            layer_type="observed_surveillance",
            data=data,
            coverage_state=(
                "observed_for_registered_sources" if eligible else "awaiting_sponsor_data"
            ),
            disease_scoped=True,
            geography_scoped=True,
            warnings=[warning],
            deferrals=deferrals,
        )

    @application.get("/api/v1/observed-surveillance/map")
    def observed_surveillance_map_read(
        request: Request,
        disease: str | None = Query(default=None, max_length=100),
        as_of: str | None = Query(default=None, max_length=10),
    ) -> Envelope[dict[str, Any]]:
        """Latest authorised district-week rate map, explicitly separate from evidence."""

        from datetime import date as date_type

        from packages.forecasting.authorised_surveillance import SurveillanceContractError
        from packages.forecasting.operational import observed_surveillance_map

        try:
            issue_date = date_type.fromisoformat(as_of) if as_of else None
        except ValueError as exc:
            raise ApiError(
                422, "INVALID_AS_OF", "invalid_as_of_date", "as_of must be YYYY-MM-DD"
            ) from exc
        try:
            data = observed_surveillance_map(disease=disease, as_of=issue_date)
        except SurveillanceContractError as exc:
            return _envelope(
                request,
                layer_type="observed_surveillance",
                data={
                    "metric": "rate_per_100k",
                    "records": [],
                    "no_data_semantics": (
                        "No authorised observation was supplied; no zero is inferred."
                    ),
                },
                coverage_state="awaiting_sponsor_data",
                disease_scoped=True,
                geography_scoped=True,
                warnings=[
                    WarningItem(
                        code="OBSERVED_SURVEILLANCE_UNAVAILABLE",
                        severity="blocking",
                        message=str(exc),
                    )
                ],
                deferrals=[
                    DeferralItem(
                        capability="observed_surveillance",
                        state=CapabilityState(code="awaiting_sponsor_data"),
                        reason_code="AUTHORISED_ROUTINE_AGGREGATES_NOT_SUPPLIED",
                    )
                ],
            )
        return _envelope(
            request,
            layer_type="observed_surveillance",
            data=data,
            coverage_state=(
                "observed_for_registered_sources"
                if all(item["observation_state"] == "observed_complete" for item in data["records"])
                else "partial"
            ),
            disease_scoped=True,
            geography_scoped=True,
            warnings=[
                WarningItem(
                    code="OBSERVED_SURVEILLANCE_RATE",
                    severity="info",
                    message=(
                        "This is an authorised observed rate map. It is neither an article "
                        "count nor a forecast; records below the completeness floor are marked."
                    ),
                )
            ],
        )

    @application.get("/api/v1/forecast/operational")
    def operational_forecast_summary(request: Request) -> Envelope[dict[str, Any]]:
        """List trained authorised models; no artefact means a typed data gate."""

        from packages.forecasting.operational import OperationalForecastError, summary

        try:
            data = summary()
        except OperationalForecastError as exc:
            return _envelope(
                request,
                layer_type="forecast",
                data={"status": "awaiting_authorised_model_build", "cells": []},
                coverage_state="awaiting_sponsor_data",
                disease_scoped=True,
                geography_scoped=True,
                warnings=[
                    WarningItem(
                        code="OPERATIONAL_FORECAST_UNAVAILABLE",
                        severity="blocking",
                        message=str(exc),
                    )
                ],
                deferrals=[
                    DeferralItem(
                        capability="authorised_district_week_surveillance_forecast",
                        state=CapabilityState(code="awaiting_sponsor_data"),
                        reason_code="AUTHORISATION_MODEL_ARTEFACT_NOT_BUILT",
                    )
                ],
            )
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
        )

    @application.get("/api/v1/forecast/operational/map")
    def operational_forecast_map(
        request: Request,
        disease: str = Query(..., min_length=1, max_length=100),
        horizon_weeks: int = Query(default=1),
    ) -> Envelope[dict[str, Any]]:
        """Current authorised threshold-exceedance probability map, if qualified."""

        from packages.forecasting.operational import (
            OperationalForecastError,
            current_probability_map,
        )

        try:
            data = current_probability_map(disease=disease, horizon_weeks=horizon_weeks)
        except (OperationalForecastError, FileNotFoundError) as exc:
            return _envelope(
                request,
                layer_type="forecast",
                data={
                    "status": "insufficient_evidence",
                    "disease": disease,
                    "horizon_weeks": horizon_weeks,
                    "districts": [],
                },
                coverage_state="unavailable",
                disease_scoped=True,
                geography_scoped=True,
                warnings=[
                    WarningItem(
                        code="OPERATIONAL_FORECAST_WITHHELD",
                        severity="blocking",
                        message=str(exc),
                    )
                ],
                deferrals=[
                    DeferralItem(
                        capability="authorised_district_week_surveillance_forecast",
                        state=CapabilityState(code="insufficient_evidence"),
                        reason_code="CURRENT_DATA_OR_MODEL_GATE_NOT_MET",
                    )
                ],
            )
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="observed_for_registered_sources",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[
                WarningItem(
                    code="QUALIFIED_OPERATIONAL_FORECAST",
                    severity="info",
                    message=(
                        "Probability is disease-specific threshold exceedance, not a case count; "
                        "inspect the registered threshold, completeness and model evaluation."
                    ),
                )
            ],
        )

    @application.post("/api/v1/translate")
    def translate_text(
        payload: TranslateRequest,
        request: Request,
    ) -> Envelope[dict[str, Any]]:
        """Translate between English, Hindi and Odia on-device.

        Odisha district names are protected through the gazetteer before decoding,
        because an unconstrained decoder renames them (Khordha became "गोरखा"), and a
        district name is the one token in this product that must survive intact.
        """

        from packages.nlp import translate as translation

        started = perf_counter()
        source = payload.source_language or translation.detect_language(payload.text)
        result = translation.translate(payload.text, source, payload.target_language)
        latency_ms = int((perf_counter() - started) * 1000)
        status = {"translated": "translated", "identity": "passthrough"}.get(
            result.state, "unavailable"
        )
        return _envelope(
            request,
            layer_type="not_applicable",
            coverage_state="not_applicable" if status != "unavailable" else "unavailable",
            data={
                # Stable client fields used by both the translation workspace and
                # per-evidence translation controls.
                "status": status,
                "source_text": payload.text,
                "translated_text": result.text if status != "unavailable" else None,
                "source_language_detected": payload.source_language is None,
                "model": result.engine,
                "pipeline": [
                    f"detect:{result.source_language}"
                    if payload.source_language is None
                    else f"declared:{result.source_language}",
                    result.engine,
                ],
                "latency_ms": latency_ms,
                "capability_code": result.reason_code,
                # Backwards-compatible low-level fields.
                "text": result.text,
                "source_language": result.source_language,
                "target_language": result.target_language,
                "state": result.state,
                "engine": result.engine,
                "reason_code": result.reason_code,
                "unresolved_terms": list(result.unresolved_terms),
                "is_synthetic": False,
            },
            deferrals=(
                []
                if status != "unavailable"
                else [
                    DeferralItem(
                        capability="on_device_translation",
                        state=CapabilityState(code="not_implemented_phase_one"),
                        reason_code=result.reason_code or "TRANSLATION_UNAVAILABLE",
                    )
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Present-day environmental favourability (Objective 3, current-capable).
    #
    # Deliberately filed under its own layer_type. It is a different quantity from
    # the historical catalogue experiment and must never be read as a case forecast.
    # ------------------------------------------------------------------

    @contextmanager
    def _environment_guard() -> Iterator[None]:
        from packages.forecasting import CurrentConditionsUnavailable

        try:
            yield
        except (CurrentConditionsUnavailable, FileNotFoundError) as exc:
            raise ApiError(
                503,
                "CURRENT_CONDITIONS_UNAVAILABLE",
                "current_conditions_layer_not_built",
                str(exc),
            ) from exc

    def _environment_envelope(
        request: Request, data: dict[str, Any], *, detail: bool
    ) -> Envelope[dict[str, Any]]:
        scored = (data.get("coverage") or {}).get("scored")
        warnings = [
            WarningItem(
                code="NOT_A_CASE_FORECAST",
                severity="blocking",
                message=(data.get("quantity") or {}).get("statement", ""),
            )
        ]
        warnings += [
            WarningItem(code="ENVIRONMENT_LAYER", severity="warning", message=str(item))
            for item in data.get("warnings", [])
        ]
        deferrals: list[DeferralItem] = []
        if detail:
            blocked = (data.get("sources") or {}).get("meteorology", {}).get("blocked_surfaces", [])
            capability_states: dict[str, CapabilityCode] = {
                "credentials_required": "awaiting_external_credential",
                "licence_acceptance_required": "awaiting_external_credential",
                "awaiting_source_permission_or_approved_api": (
                    "awaiting_source_permission_or_approved_api"
                ),
                "provider_unavailable": "source_temporarily_unavailable",
            }
            deferrals = [
                DeferralItem(
                    capability=f"imd_gateway:{item.get('product')}",
                    state=CapabilityState(
                        code=capability_states.get(
                            str(item.get("state")),
                            "source_temporarily_unavailable",
                        )
                    ),
                    reason_code="IMD_GATEWAY_REQUIRES_REGISTRATION",
                )
                for item in blocked
            ]
        return _envelope(
            request,
            layer_type="environment",
            data=data,
            coverage_state=("observed_for_registered_sources" if scored == 30 else "partial"),
            geography_scoped=True,
            warnings=warnings,
            deferrals=deferrals,
        )

    @application.get("/api/v1/environment/current/map")
    def environment_current_map(request: Request) -> Envelope[dict[str, Any]]:
        from packages.forecasting import current_conditions_map

        with _environment_guard():
            data = current_conditions_map()
        return _environment_envelope(request, data, detail=False)

    @application.get("/api/v1/environment/current")
    def environment_current_detail(request: Request) -> Envelope[dict[str, Any]]:
        from packages.forecasting import current_conditions_layer

        with _environment_guard():
            data = current_conditions_layer()
        return _environment_envelope(request, data, detail=True)

    # ------------------------------------------------------------------
    # Authoritative public observed layers and public-data research outlook.
    # These are available without sponsor credentials and are the primary
    # disease-map/predictive-analysis surfaces of the hackathon deployment.
    # ------------------------------------------------------------------

    @application.get("/api/v1/public-health/malaria/map")
    def public_malaria_map(
        request: Request,
        year: int | None = Query(default=None, ge=2010, le=2024),
        metric: str = Query(default="api", max_length=40),
    ) -> Envelope[dict[str, Any]]:
        from .public_health import PublicHealthDataError, malaria_map

        try:
            data = malaria_map(year=year, metric=metric)
        except ValueError as exc:
            raise ApiError(422, "INVALID_PUBLIC_MAP_QUERY", "invalid_map_query", str(exc)) from exc
        except PublicHealthDataError as exc:
            raise ApiError(
                503, "PUBLIC_DISEASE_DATA_UNAVAILABLE", "public_disease_data_unavailable", str(exc)
            ) from exc
        warnings = [
            WarningItem(
                code="ANNUAL_NOT_CURRENT_WEEK",
                severity="info",
                message=(
                    "Official NCVBDC annual district malaria statistics. The map is "
                    "observed annual burden, not a current district-week incidence map."
                ),
            )
        ]
        # A year whose source table never printed the requested column renders as a
        # blank map. Silence there reads as "no malaria"; it means "not published".
        records = data["records"]
        if records and all(
            item["observation_state"] == "not_reported_in_table" for item in records
        ):
            warnings.append(
                WarningItem(
                    code="METRIC_NOT_PRINTED_IN_SOURCE_YEAR",
                    severity="warning",
                    message=(
                        f"The official {data['year']} NCVBDC table does not print "
                        f"'{data['metric']}' for any Odisha district, so every district is "
                        "blank because the value was never published — not because it is "
                        "zero or low. Try another metric or year."
                    ),
                )
            )
        return _envelope(
            request,
            layer_type="observed_surveillance",
            data=data,
            coverage_state=(
                "observed_for_registered_sources"
                if any(item["observation_state"] == "observed" for item in records)
                else "unavailable"
            ),
            disease_scoped=True,
            geography_scoped=True,
            warnings=warnings,
        )

    @application.get("/api/v1/public-health/hmis/map")
    def public_hmis_map(
        request: Request,
        period: str | None = Query(default=None, max_length=10),
        metric: str = Query(default="malaria_test_positivity", max_length=60),
    ) -> Envelope[dict[str, Any]]:
        from .public_health import PublicHealthDataError, hmis_map

        try:
            if period is not None:
                datetime.strptime(period, "%Y-%m-%d")
            data = hmis_map(period=period, metric=metric)
        except ValueError as exc:
            raise ApiError(422, "INVALID_PUBLIC_MAP_QUERY", "invalid_map_query", str(exc)) from exc
        except PublicHealthDataError as exc:
            raise ApiError(
                503, "PUBLIC_DISEASE_DATA_UNAVAILABLE", "public_disease_data_unavailable", str(exc)
            ) from exc
        return _envelope(
            request,
            layer_type="observed_surveillance",
            data=data,
            coverage_state="observed_for_registered_sources",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[
                WarningItem(
                    code="HMIS_RECORDS_NOT_INCIDENCE",
                    severity="warning",
                    message=(
                        "HMIS values are provisional facility-reported test/service records; "
                        "they are not deduplicated people or population incidence."
                    ),
                )
            ],
        )

    @application.get("/api/v1/outlook/public/map")
    def public_research_outlook_map(
        request: Request,
        disease: str = Query(default="malaria", max_length=40),
        horizon_month: int = Query(default=1, ge=1, le=3),
    ) -> Envelope[dict[str, Any]]:
        from .public_health import PublicHealthDataError, public_outlook_map

        if disease.casefold() != "malaria":
            raise ApiError(
                422,
                "PUBLIC_OUTLOOK_DISEASE_UNSUPPORTED",
                "only_malaria_has_sufficient_public_training_data",
                "The public research outlook currently supports malaria only.",
            )
        try:
            data = public_outlook_map(horizon_month=horizon_month)
        except (ValueError, PublicHealthDataError) as exc:
            raise ApiError(
                503, "PUBLIC_OUTLOOK_UNAVAILABLE", "public_outlook_unavailable", str(exc)
            ) from exc
        # The outlook serves a forward window from a model whose training panel
        # stops in 2020, so the envelope carries that panel end as its vintage
        # rather than `not_applicable`. The tagged Scoped value cannot hold a value
        # and a reason code at once, so the reason code travels as a warning.
        vintage = data["training_data_vintage"]
        warnings = [
            WarningItem(
                code="RESEARCH_INDICATOR_OUTLOOK_NOT_OUTBREAK_PROBABILITY",
                severity="warning",
                message=(
                    "This is a 1-3 month public-data research outlook for elevated "
                    "HMIS microscopy positivity and surveillance priority. It is not "
                    "a calibrated probability of a disease outbreak or an alert."
                ),
            ),
            WarningItem(
                code=str(vintage["reason_code"]),
                severity="warning",
                message=str(vintage["detail"]),
            ),
            WarningItem(
                code=(
                    "MODEL_HAS_SKILL_OVER_UNCONDITIONAL_CLIMATOLOGY"
                    if data["beats_unconditional_climatology"]
                    else "NO_MODEL_BEAT_UNCONDITIONAL_CLIMATOLOGY"
                ),
                severity="info" if data["beats_unconditional_climatology"] else "warning",
                message=str(data["model_skill_statement"]),
            ),
        ]
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
            data_vintage=Scoped[DataVintage].present(
                DataVintage(
                    vintage_id=(
                        "hmis_monthly_panel_end_"
                        f"{vintage['hmis_training_series_end']}"
                    )
                )
            ),
            warnings=warnings,
        )

    @application.get("/api/v1/outlook/public/evaluation")
    def public_research_outlook_evaluation(request: Request) -> Envelope[dict[str, Any]]:
        from .public_health import PublicHealthDataError, public_outlook_evaluation

        try:
            data = public_outlook_evaluation()
        except PublicHealthDataError as exc:
            raise ApiError(
                503, "PUBLIC_OUTLOOK_UNAVAILABLE", "public_outlook_unavailable", str(exc)
            ) from exc
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
        )

    # ------------------------------------------------------------------
    # Historical EpiClim catalogue-row experiment (Objective 3 research surface).
    #
    # These read a committed, pre-fitted artefact. Nothing is ever fitted inside a
    # request. The modelled quantity is whether the frozen, incomplete EpiClim file
    # contains a matching row dated in a district-week. It is not incidence, an
    # official-publication probability or an operational outbreak forecast. The
    # separate /api/v1/forecast route continues to refuse incidence outright.
    # ------------------------------------------------------------------

    def _forecast_group(value: str) -> str:
        if value not in _REAL_FORECAST_GROUPS:
            raise ApiError(
                422,
                "UNSUPPORTED_DISEASE_GROUP",
                "disease_group_not_modelled",
                "disease_group must be one of " + ", ".join(sorted(_REAL_FORECAST_GROUPS)) + ".",
            )
        return value

    def _forecast_horizon(value: int) -> int:
        if value not in _REAL_FORECAST_HORIZONS:
            raise ApiError(
                422,
                "UNSUPPORTED_FORECAST_HORIZON",
                "horizon_must_be_1_2_4_8_or_12_weeks",
                "horizon_weeks must be one of 1, 2, 4, 8 or 12.",
            )
        return value

    @application.get("/api/v1/forecast/real")
    def real_forecast_summary(request: Request) -> Envelope[dict[str, Any]]:
        from packages.forecasting import summary as _summary

        with _forecast_artefact_guard():
            data = _summary()
        deferrals = [
            DeferralItem(
                capability=(
                    "epiclim_catalogue_row_experiment:"
                    f"{cell['disease_group']}:{cell['horizon_weeks']}w"
                ),
                state=CapabilityState(code="insufficient_evidence"),
                reason_code=cell["reason_codes"][0],
            )
            for cell in data.get("refused_cells", [])
        ]
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[_not_incidence_warning()],
            deferrals=deferrals,
        )

    @application.get("/api/v1/forecast/real/map")
    def real_forecast_map(
        request: Request,
        disease_group: str = Query(default="any_reported_outbreak", max_length=60),
        horizon_weeks: int = Query(default=1),
    ) -> Envelope[dict[str, Any]]:
        from packages.forecasting import probability_map

        group = _forecast_group(disease_group)
        horizon = _forecast_horizon(horizon_weeks)
        with _forecast_artefact_guard():
            payload = probability_map(group, horizon)
        if payload.get("status") == "experimental":
            return _envelope(
                request,
                layer_type="forecast",
                data=payload,
                coverage_state="partial",
                disease_scoped=True,
                geography_scoped=True,
                warnings=[
                    _not_incidence_warning(),
                    WarningItem(
                        code="HISTORICAL_REISSUE",
                        severity="blocking",
                        message=(
                            f"Indexed at {payload['issue_week']}; the frozen EpiClim "
                            "series ends 2022-12-31, so this is a retrospective "
                            "dataset-membership experiment, not current disease risk."
                        ),
                    ),
                ],
            )
        return _envelope(
            request,
            layer_type="forecast",
            data=payload,
            coverage_state="unavailable",
            disease_scoped=True,
            geography_scoped=True,
            data_vintage=Scoped[DataVintage].unavailable("MODEL_DID_NOT_BEAT_SEASONAL_BASELINE"),
            warnings=[_not_incidence_warning()],
            deferrals=[
                DeferralItem(
                    capability="epiclim_catalogue_row_experiment",
                    state=CapabilityState(code="insufficient_evidence"),
                    reason_code=payload["reason_codes"][0],
                )
            ],
        )

    @application.get("/api/v1/forecast/real/evaluation")
    def real_forecast_evaluation(
        request: Request,
        disease_group: str = Query(default="any_reported_outbreak", max_length=60),
        horizon_weeks: int = Query(default=1),
    ) -> Envelope[dict[str, Any]]:
        from packages.forecasting import evaluation

        group = _forecast_group(disease_group)
        horizon = _forecast_horizon(horizon_weeks)
        with _forecast_artefact_guard():
            data = evaluation(group, horizon)
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[_not_incidence_warning()],
        )

    @application.get("/api/v1/forecast/real/current")
    def real_forecast_current(request: Request) -> Envelope[dict[str, Any]]:
        from packages.forecasting import current_week_refusal

        with _forecast_artefact_guard():
            data = current_week_refusal()
        return _envelope(
            request,
            layer_type="forecast",
            data=data,
            coverage_state="awaiting_sponsor_data",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[_not_incidence_warning()],
            deferrals=[
                DeferralItem(
                    capability="epiclim_catalogue_row_experiment_current_week",
                    state=CapabilityState(code="public_catalogue_only_no_denominator"),
                    reason_code="TARGET_SERIES_ENDS_BEFORE_REQUESTED_ISSUE_DATE",
                )
            ],
        )

    @application.get("/api/v1/audits/epiclim")
    def epiclim_audit(request: Request) -> Envelope[dict[str, Any]]:
        audit = _load_epiclim_audit()
        return _envelope(
            request,
            layer_type="official_event_catalogue",
            data=audit,
            coverage_state="partial",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[
                WarningItem(
                    code="NOT_ROUTINE_SURVEILLANCE",
                    severity="blocking",
                    message=(
                        "This positive-only catalogue must not be reindexed with "
                        "missing weeks as zero cases."
                    ),
                )
            ],
        )

    @application.post("/api/v1/demo/replay-fixtures")
    def replay_fixtures(
        request: Request,
        _: None = Depends(require_demo_token),
    ) -> Envelope[dict[str, Any]]:
        return _envelope(
            request,
            layer_type="public_source_signal",
            data=database.replay_demo_fixtures(),
            coverage_state="fixture_fallback",
            disease_scoped=True,
            geography_scoped=True,
            warnings=[
                WarningItem(
                    code="SYNTHETIC_FIXTURES",
                    severity="info",
                    message="These workflow records are visibly synthetic fixtures.",
                )
            ],
        )

    @application.post("/api/v1/demo/synthetic-forecast/run")
    def synthetic_forecast(
        payload: SyntheticForecastRequest,
        request: Request,
        _: None = Depends(require_demo_token),
    ) -> Envelope[dict[str, Any]]:
        result = build_synthetic_report(payload.seed, payload.horizon_weeks)
        return _envelope(
            request,
            layer_type="forecast",
            data=result,
            coverage_state="fixture_fallback",
            disease_scoped=True,
            geography_scoped=False,
            warnings=[
                WarningItem(
                    code=WATERMARK,
                    severity="blocking",
                    message="Synthetic software test; this is not an Odisha disease-risk estimate.",
                )
            ],
            deferrals=[
                DeferralItem(
                    capability="forecast_harness",
                    state=CapabilityState(code="simulation_only_not_odisha_risk"),
                    reason_code=WATERMARK,
                )
            ],
        )

    @application.get("/api/v1/demo/synthetic-forecast")
    def synthetic_forecast_read(
        request: Request,
        horizon_weeks: int = Query(default=12, ge=1, le=12),
    ) -> Envelope[dict[str, Any]]:
        """Public, deterministic software harness; never an Odisha risk estimate."""

        if horizon_weeks not in {1, 2, 4, 8, 12}:
            raise ApiError(
                422,
                "UNSUPPORTED_SYNTHETIC_HORIZON",
                "synthetic_horizon_must_be_1_2_4_8_or_12_weeks",
                "Synthetic horizon must be one of 1, 2, 4, 8 or 12 weeks.",
            )
        result = build_synthetic_report(20260721, horizon_weeks)
        return _envelope(
            request,
            layer_type="forecast",
            data=result,
            coverage_state="fixture_fallback",
            disease_scoped=True,
            geography_scoped=False,
            warnings=[
                WarningItem(
                    code=WATERMARK,
                    severity="blocking",
                    message="Synthetic software test; this is not an Odisha disease-risk estimate.",
                )
            ],
            deferrals=[
                DeferralItem(
                    capability="forecast_harness",
                    state=CapabilityState(code="simulation_only_not_odisha_risk"),
                    reason_code=WATERMARK,
                )
            ],
        )

    @application.post("/api/v1/forecast/run")
    def real_forecast_run(request: Request) -> None:
        raise ApiError(
            501,
            "TARGET_SERIES_INELIGIBLE",
            "insufficient_training_data",
            (
                "No revision-aware routine Odisha surveillance target with denominators "
                "and sufficient independent events has been authorised."
            ),
        )

    @application.post("/api/v1/internal/jobs/enqueue")
    def enqueue_job(
        payload: EnqueueJobRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        key = _idempotency_key(idempotency_key)
        try:
            job, replayed = database.enqueue_job(
                source_id=payload.source_id,
                kind=payload.kind,
                payload_ref=payload.payload_ref,
                payload_hash=payload.payload_hash,
                idempotency_key=key,
            )
        except (RepositoryConflict, RepositoryNotFound) as error:
            raise translate_repository_error(error) from error
        return _envelope(
            request, layer_type="not_applicable", data={"job": job, "idempotent_replay": replayed}
        )

    @application.post("/api/v1/internal/jobs/claim")
    def claim_job(
        payload: JobClaimRequest,
        request: Request,
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        try:
            job = database.claim_job(owner=payload.owner, lease_seconds=payload.lease_seconds)
        except RepositoryConflict as error:
            raise translate_repository_error(error) from error
        return _envelope(request, layer_type="not_applicable", data={"job": job})

    @application.post("/api/v1/internal/jobs/{job_id}/complete")
    def complete_job(
        job_id: str,
        payload: CompleteJobRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        key = _idempotency_key(idempotency_key)
        try:
            job, replayed, task_ids = database.complete_job(
                job_id=job_id,
                owner=payload.owner,
                fencing_token=payload.fencing_token,
                idempotency_key=key,
                receipt=payload.receipt,
                signals=payload.signals,
            )
        except (RepositoryConflict, RepositoryNotFound) as error:
            raise translate_repository_error(error) from error
        return _envelope(
            request,
            layer_type="not_applicable",
            data={"job": job, "review_task_ids": task_ids, "idempotent_replay": replayed},
        )

    @application.post("/api/v1/internal/jobs/{job_id}/fail")
    def fail_job(
        job_id: str,
        payload: FailJobRequest,
        request: Request,
        _: None = Depends(require_collector_token),
    ) -> Envelope[dict[str, Any]]:
        try:
            job = database.fail_job(
                job_id=job_id,
                owner=payload.owner,
                fencing_token=payload.fencing_token,
                reason_code=payload.reason_code,
                retryable=payload.retryable,
            )
        except (RepositoryConflict, RepositoryNotFound) as error:
            raise translate_repository_error(error) from error
        return _envelope(request, layer_type="not_applicable", data={"job": job})

    return application


app = create_app()
