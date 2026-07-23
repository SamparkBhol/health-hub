from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from packages.contracts.api import (
    LIVE_EVIDENCE_PLACEHOLDER,
    LIVE_EVIDENCE_REDACTION_STATE,
    RedactedSignalInput,
    SourceReceiptInput,
)
from services.api.main import create_app
from workers.ingestion.registry import load_registry


def make_client(tmp_path) -> TestClient:
    app = create_app(f"sqlite:///{tmp_path / 'api.sqlite3'}")
    return TestClient(app)


def assert_envelope(payload: dict) -> None:
    assert payload["schema_version"] == "1.0.0"
    assert payload["request_id"].startswith("req_")
    assert payload["deployment_profile"] == "hackathon_production_shaped"
    assert payload["context"]["as_of"]["state"] == "value"
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["deferrals"], list)


def test_public_contracts_and_audit_are_explicit(tmp_path) -> None:
    client = make_client(tmp_path)

    assert client.get("/api/v1/healthz").json() == {
        "status": "alive",
        "scope": "process_liveness_only",
    }

    readiness = client.get("/api/v1/readiness")
    assert readiness.status_code == 200
    assert_envelope(readiness.json())
    capabilities = {item["capability"]: item for item in readiness.json()["data"]["capabilities"]}
    assert capabilities["official_public_disease_maps"]["state"]["code"] == "available"
    assert capabilities["public_three_month_research_outlook"]["state"]["code"] == (
        "research_only_not_operational_alert"
    )
    assert capabilities["authorised_operational_outbreak_forecast"]["state"]["code"] == (
        "target_series_ineligible"
    )
    assert capabilities["district_geometry"]["state"]["code"] == "community_demo_boundary"

    sources = client.get("/api/v1/sources").json()
    assert_envelope(sources)
    # Every registered route is published, however many the registry holds.
    assert len(sources["data"]) == len(load_registry().sources)
    idsp = next(item for item in sources["data"] if item["id"] == "idsp_weekly_outbreaks")
    assert idsp["state"] == "registered_uncontacted"
    assert "Wayback" in idsp["note"]

    empty_signals = client.get("/api/v1/signals").json()
    assert empty_signals["data"] == []
    assert empty_signals["context"]["coverage_state"] == "unknown"
    assert any(
        item["code"] == "NO_SUCCESSFUL_COLLECTION_RECEIPT" for item in empty_signals["warnings"]
    )

    audit_response = client.get("/api/v1/audits/epiclim")
    assert audit_response.status_code == 200
    audit = audit_response.json()
    assert_envelope(audit)
    assert audit["data"]["odisha"]["rows"] == 358
    assert audit["data"]["odisha"]["disease_counts"]["Dengue"] == 2
    assert audit["data"]["national"]["week_index_mismatch_gt_one_week_rows"] == 2517
    assert audit["data"]["eligibility"]["district_week_count_forecast"] == "ineligible"


def test_root_returns_manifest_or_redirects_to_configured_web_app(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WEB_APP_URL", raising=False)
    client = make_client(tmp_path)
    manifest = client.get("/")
    assert manifest.status_code == 200
    assert manifest.json() == {
        "name": "Janaswasthya Agentic Public Health Intelligence API",
        "status": "alive",
        "web_url": None,
        "documentation": "/docs",
        "health": "/api/v1/healthz",
    }

    monkeypatch.setenv("WEB_APP_URL", "http://localhost:5173")
    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://localhost:5173"


def test_readiness_is_503_when_database_is_unavailable(tmp_path, monkeypatch) -> None:
    client = make_client(tmp_path)
    monkeypatch.setattr(client.app.state.database, "ready", lambda: False)
    response = client.get("/api/v1/readyz")
    assert response.status_code == 503
    assert response.json()["data"]["ready"] is False
    assert response.json()["data"]["database"] == "unavailable"
    assert client.get("/api/v1/healthz").status_code == 200


def test_cors_is_explicit_not_wildcard(tmp_path) -> None:
    client = make_client(tmp_path)
    allowed = client.options(
        "/api/v1/readiness",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.get("/api/v1/readiness", headers={"Origin": "https://attacker.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_api_responses_apply_browser_security_and_default_no_store_headers(
    tmp_path,
) -> None:
    response = make_client(tmp_path).get("/api/v1/readiness")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == ("camera=(), microphone=(), geolocation=()")
    assert response.headers["strict-transport-security"].startswith("max-age=31536000")
    assert response.headers["cache-control"] == "no-store"


def test_pending_pdf_operator_view_is_authenticated_and_never_exposes_url(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("COLLECTOR_API_TOKEN", "collector-secret")
    client = make_client(tmp_path)
    database = client.app.state.database
    source_id = "odisha_hfw_circulars_en"
    url = "https://health.odisha.gov.in/reports/patient-name.pdf"
    database.register_discovered_links(
        source_id=source_id,
        links=[
            {
                "url": url,
                "label": "private-looking anchor label",
                "content_hint": "application/pdf",
            }
        ],
    )
    reserved = database.reserve_discovered_links(source_id=source_id, limit=1)[0]
    job, _ = database.enqueue_reserved_discovered_link(source_id=source_id, url=reserved["url"])
    claimed = database.claim_job(
        owner="operator-view-test",
        lease_seconds=300,
        kind="fetch",
        payload_prefix="registered-link:",
    )
    assert claimed is not None and claimed["id"] == job["id"]
    observed_digest = "d" * 64
    database.complete_job(
        job_id=claimed["id"],
        owner="operator-view-test",
        fencing_token=claimed["fencing_token"],
        idempotency_key="operator-view-complete",
        receipt=SourceReceiptInput(
            source_snapshot_id="snapshot_pending_operator_view",
            source_id=source_id,
            requested_url=url,
            final_url=url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="application/pdf",
            byte_length=100,
            sha256=observed_digest,
            access_path="live_origin",
        ),
        signals=[],
        link_disposition="pending_approval",
    )

    denied = client.get("/api/v1/internal/collector/pending-pdfs")
    assert denied.status_code == 401
    response = client.get(
        "/api/v1/internal/collector/pending-pdfs",
        headers={"X-Collector-Token": "collector-secret"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["count"] == 1
    item = payload["data"]["items"][0]
    assert item["observed_content_sha256"] == observed_digest
    assert item["url_sha256"] == hashlib.sha256(url.encode()).hexdigest()
    serialized = response.text
    assert url not in serialized
    assert "private-looking anchor label" not in serialized
    missing_actor = client.get(
        "/api/v1/internal/collector/pending-pdfs",
        params={"include_inspection_url": "true"},
        headers={"X-Collector-Token": "collector-secret"},
    )
    assert missing_actor.status_code == 400
    protected = client.get(
        "/api/v1/internal/collector/pending-pdfs",
        params={"include_inspection_url": "true"},
        headers={
            "X-Collector-Token": "collector-secret",
            "X-Operator-ID": "operator-a",
        },
    )
    assert protected.status_code == 200
    assert protected.json()["data"]["items"][0]["inspection_url"] == url


def test_boundary_is_real_pinned_geojson_with_attribution(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/v1/boundaries/districts")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    assert response.headers["x-boundary-authority"] == "community_demo"
    assert response.headers["x-boundary-vintage"] == "Census 2011"
    assert len(response.headers["x-boundary-sha256"]) == 64
    assert 'rel="license"' in response.headers["link"]
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 30
    district_ids = {item["properties"]["district_id"] for item in geojson["features"]}
    assert "OD-DIST-khordha" in district_ids
    assert all(
        item["properties"]["boundary_authority"] == "community_demo" for item in geojson["features"]
    )


def test_forecast_boundary_is_machine_enforced(tmp_path) -> None:
    client = make_client(tmp_path)

    read = client.get("/api/v1/forecast")
    assert read.status_code == 200
    payload = read.json()
    assert_envelope(payload)
    assert payload["data"] == []
    assert payload["context"]["coverage_state"] == "awaiting_sponsor_data"
    assert payload["context"]["data_vintage"]["state"] == "unavailable"
    assert payload["warnings"][0]["code"] == "NOT_A_FORECAST"

    refused = client.post("/api/v1/forecast/run")
    assert refused.status_code == 501
    problem = refused.json()
    assert_envelope(problem)
    assert problem["data"]["code"] == "TARGET_SERIES_INELIGIBLE"
    assert problem["data"]["reason_code"] == "insufficient_training_data"

    synthetic = client.post("/api/v1/demo/synthetic-forecast/run", json={})
    assert synthetic.status_code == 200
    result = synthetic.json()
    assert_envelope(result)
    assert result["data"]["watermark"] == "SIMULATION_ONLY_NOT_ODISHA_RISK"
    assert result["data"]["is_synthetic"] is True
    assert result["data"]["real_odisha_prediction_available"] is False
    assert len(result["data"]["latest_simulation_map"]) == 30
    assert all(
        item["watermark"] == "SIMULATION_ONLY_NOT_ODISHA_RISK"
        for item in result["data"]["latest_simulation_map"]
    )

    three_month = client.post(
        "/api/v1/demo/synthetic-forecast/run", json={"horizon_weeks": 12}
    ).json()
    assert three_month["data"]["horizon_weeks"] == 12
    assert three_month["data"]["real_odisha_prediction_available"] is False
    public_harness = client.get("/api/v1/demo/synthetic-forecast?horizon_weeks=12").json()
    assert public_harness["data"]["watermark"] == "SIMULATION_ONLY_NOT_ODISHA_RISK"
    assert len(public_harness["data"]["latest_simulation_map"]) == 30


def test_published_signal_map_is_filtered_server_side_and_never_incidence(tmp_path) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    default_signals = client.get("/api/v1/signals").json()
    assert default_signals["data"]
    assert all(item["assertion"] == "affirmed" for item in default_signals["data"])
    all_signals = client.get("/api/v1/signals", params={"assertion": "all"}).json()
    assert any(item["assertion"] != "affirmed" for item in all_signals["data"])
    default_map = client.get("/api/v1/maps/published-signals").json()
    all_mentions_map = client.get(
        "/api/v1/maps/published-signals", params={"assertion": "all"}
    ).json()
    assert default_map["data"]["filters"]["assertion"] == "affirmed"
    assert sum(item["published_signal_count"] for item in default_map["data"]["districts"]) < sum(
        item["published_signal_count"] for item in all_mentions_map["data"]["districts"]
    )
    response = client.get(
        "/api/v1/maps/published-signals",
        params={"disease": "dengue", "assertion": "affirmed", "language": "or"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["metric"] == "published_signal_count"
    assert payload["data"]["time_axis"] == "retrieval_time_not_event_onset"
    assert payload["data"]["fixture_mode"] == "fixture_only"
    assert payload["context"]["coverage_state"] == "fixture_fallback"
    assert all(row["published_signal_count"] > 0 for row in payload["data"]["districts"])
    assert any(item["code"] == "NOT_DISEASE_INCIDENCE" for item in payload["warnings"])
    offset = client.get(
        "/api/v1/maps/published-signals",
        params={"retrieved_from": "2026-07-21T17:00:00+05:30"},
    ).json()
    assert offset["data"]["filters"]["retrieved_from"] == "2026-07-21T11:30:00Z"
    invalid = client.get(
        "/api/v1/maps/published-signals",
        params={
            "retrieved_from": "2026-07-22T00:00:00Z",
            "retrieved_to": "2026-07-21T00:00:00Z",
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["data"]["code"] == "INVALID_TIME_RANGE"


def test_generic_public_signal_layer_uses_privacy_safe_serializer(tmp_path) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    response = client.get("/api/v1/layers/public_source_signal")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]
    assert all(item["assertion"] == "affirmed" for item in payload["data"])
    assert all(
        item["canonicalUrlState"] == "registered_source_only_detail_url_withheld"
        for item in payload["data"]
    )
    serialized = response.text
    assert '"requested_url"' not in serialized
    assert '"final_url"' not in serialized
    assert "fixtures.invalid" not in serialized


def test_quality_hold_is_internal_only_and_cannot_drive_public_map_or_agent(
    tmp_path,
) -> None:
    client = make_client(tmp_path)
    database = client.app.state.database
    source_id = "ganjam_collectorate"
    source_url = "https://ganjam.odisha.gov.in/or/health-update"
    job, _ = database.enqueue_job(
        source_id=source_id,
        kind="fetch",
        payload_ref=f"registered-link:{source_url}",
        payload_hash="a" * 64,
        idempotency_key="privacy-hold-enqueue",
    )
    claimed = database.claim_job(owner="privacy-hold-test", lease_seconds=300, job_id=job["id"])
    assert claimed is not None
    evidence = "जुएल ओराम और झिंगिया ओराम से जुड़ा डेंगू उल्लेख गंजाम में प्रकाशित हुआ।"
    evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()
    database.complete_job(
        job_id=claimed["id"],
        owner="privacy-hold-test",
        fencing_token=claimed["fencing_token"],
        idempotency_key="privacy-hold-complete",
        receipt=SourceReceiptInput(
            source_snapshot_id="snapshot_privacy_hold",
            source_id=source_id,
            requested_url=source_url,
            final_url=source_url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="text/html",
            byte_length=len(evidence.encode()),
            sha256="b" * 64,
            access_path="live_origin",
        ),
        signals=[
            RedactedSignalInput(
                source_id=source_id,
                source_snapshot_id="snapshot_privacy_hold",
                district_id="OD-DIST-ganjam",
                disease="dengue",
                assertion="affirmed",
                evidence_text=evidence,
                evidence_start=0,
                evidence_end=len(evidence),
                content_sha256=evidence_hash,
                retrieved_at="2026-07-21T14:00:00Z",
                processing_state="privacy_review_required",
                language="hi",
            )
        ],
    )

    public_signals = client.get("/api/v1/signals").text
    public_map = client.get("/api/v1/maps/published-signals").text
    public_layer = client.get("/api/v1/layers/public_source_signal").text
    assistant = client.post(
        "/api/v1/agent/query",
        json={"question": "गंजाम में डेंगू की सूचना दिखाइए"},
    ).text
    for public_response in (public_signals, public_map, public_layer, assistant):
        assert "जुएल ओराम" not in public_response
        assert "झिंगिया ओराम" not in public_response
    assert client.get("/api/v1/signals").json()["data"] == []
    assert client.get("/api/v1/maps/published-signals").json()["data"]["districts"] == []

    protected_tasks = client.get("/api/v1/review/tasks").json()["data"]
    held = next(item for item in protected_tasks if item["source_id"] == source_id)
    assert held["task_kind"] == "quality_hold"
    assert held["processing_state"] == "privacy_review_required"
    assert held["evidence_text"] == LIVE_EVIDENCE_PLACEHOLDER
    assert held["redaction_state"] == LIVE_EVIDENCE_REDACTION_STATE
    assert held["registered_source_url"].startswith("https://ganjam.odisha.gov.in/")
    assert "जुएल ओराम" not in str(held)
    assert "झिंगिया ओराम" not in str(held)


def test_live_evidence_span_is_masked_even_when_processing_state_is_active(
    tmp_path,
) -> None:
    client = make_client(tmp_path)
    database = client.app.state.database
    source_id = "ganjam_collectorate"
    source_url = "https://ganjam.odisha.gov.in/or/active-evidence"
    job, _ = database.enqueue_job(
        source_id=source_id,
        kind="fetch",
        payload_ref=f"registered-link:{source_url}",
        payload_hash="c" * 64,
        idempotency_key="active-mask-enqueue",
    )
    claimed = database.claim_job(owner="active-mask-test", lease_seconds=300, job_id=job["id"])
    assert claimed is not None
    evidence = "Potentially identifying live evidence: dengue in Ganjam."
    database.complete_job(
        job_id=claimed["id"],
        owner="active-mask-test",
        fencing_token=claimed["fencing_token"],
        idempotency_key="active-mask-complete",
        receipt=SourceReceiptInput(
            source_snapshot_id="snapshot_active_mask",
            source_id=source_id,
            requested_url=source_url,
            final_url=source_url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="text/html",
            byte_length=len(evidence.encode()),
            sha256="d" * 64,
            access_path="live_origin",
        ),
        signals=[
            RedactedSignalInput(
                source_id=source_id,
                source_snapshot_id="snapshot_active_mask",
                district_id="OD-DIST-ganjam",
                disease="dengue",
                assertion="affirmed",
                evidence_text=evidence,
                evidence_start=0,
                evidence_end=len(evidence),
                content_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
                retrieved_at="2026-07-21T14:00:00Z",
                event_review_eligible=True,
                processing_state="active_direct",
                language="en",
            )
        ],
    )

    signals = client.get("/api/v1/signals").json()["data"]
    assert len(signals) == 1
    assert signals[0]["evidence"] == LIVE_EVIDENCE_PLACEHOLDER
    assert signals[0]["evidenceVisibility"] == LIVE_EVIDENCE_REDACTION_STATE
    assert signals[0]["redactionState"] == LIVE_EVIDENCE_REDACTION_STATE
    assert evidence not in client.get("/api/v1/signals").text
    answer = client.post(
        "/api/v1/agent/query",
        json={"question": "Show dengue evidence in Ganjam"},
    ).json()["data"]
    assert answer["evidence"][0]["redacted_evidence"] is None
    assert answer["evidence"][0]["evidence_visibility"] == LIVE_EVIDENCE_REDACTION_STATE
    assert answer["evidence"][0]["redaction_state"] == LIVE_EVIDENCE_REDACTION_STATE


def test_index_receipt_does_not_masquerade_as_live_evidence_or_hide_fixtures(tmp_path) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    assert client.get("/api/v1/signals").json()["context"]["coverage_state"] == ("fixture_fallback")
    client.app.state.database.mark_source_collection("odisha_hfw_circulars_en", succeeded=True)
    signals = client.get("/api/v1/signals").json()
    assert signals["data"]
    assert all(item["isFixture"] for item in signals["data"])
    assert signals["context"]["coverage_state"] == "fixture_fallback"
    map_payload = client.get("/api/v1/maps/published-signals").json()
    assert map_payload["data"]["fixture_mode"] == "fixture_only"
    assert map_payload["data"]["districts"]
    answer = client.post(
        "/api/v1/agent/query",
        json={"question": "Show dengue evidence in Ganjam"},
    ).json()["data"]
    assert answer["answer_state"] == "records_returned"
    assert all(item["is_fixture"] for item in answer["evidence"])


def test_validation_errors_use_the_common_problem_envelope(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/internal/jobs/enqueue",
        json={
            "source_id": "odisha_hfw_circulars_en",
            "kind": "discover",
            "payload_ref": "https://attacker.example/arbitrary",
            "payload_hash": "a" * 64,
        },
        headers={"Idempotency-Key": "invalid-payload"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert_envelope(payload)
    assert payload["data"]["reason_code"] == "request_validation_failed"


def test_translation_endpoint_matches_the_web_client_contract(tmp_path, monkeypatch) -> None:
    from packages.nlp import translate as translation

    monkeypatch.setattr(translation, "detect_language", lambda _text: "en")
    monkeypatch.setattr(
        translation,
        "translate",
        lambda text, source, target: translation.TranslationResult(
            text="ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁ ସୂଚନା",
            source_language=source,
            target_language=target,
            state="translated",
            engine="test-indictrans2",
        ),
    )
    response = make_client(tmp_path).post(
        "/api/v1/translate",
        json={"text": "Dengue notice in Khordha", "target_language": "or"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["coverage_state"] == "not_applicable"
    assert payload["data"]["status"] == "translated"
    assert payload["data"]["translated_text"] == "ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁ ସୂଚନା"
    assert payload["data"]["source_language_detected"] is True
    assert payload["data"]["model"] == "test-indictrans2"
    assert payload["data"]["pipeline"] == ["detect:en", "test-indictrans2"]


def test_operational_forecast_readiness_exposes_the_no_pii_data_contract(tmp_path) -> None:
    payload = make_client(tmp_path).get("/api/v1/forecast/operational/readiness").json()
    assert payload["context"]["layer_type"] == "observed_surveillance"
    assert payload["data"]["eligible_for_training"] is False
    assert payload["data"]["status"] == "awaiting_authorised_aggregate_export"
    assert "known_at" in payload["data"]["required_columns"]


def test_evidence_agent_understands_three_scripts_and_refuses_invention(tmp_path) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")

    odia = client.post(
        "/api/v1/agent/query",
        json={"question": "ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁ ସମ୍ଭାବନା ଆଗାମୀ ମାସରେ କେତେ?"},
    )
    assert odia.status_code == 200
    assert odia.json()["data"]["intent"] == "forecast_request"
    assert odia.json()["data"]["answer_state"] == "risk_factor_outlook"
    assert odia.json()["data"]["evidence"] == []
    assert odia.json()["data"]["outlook"]["is_synthetic"] is False

    hindi = client.post(
        "/api/v1/agent/query",
        json={"question": "गंजाम में डेंगू की चेतावनी दिखाइए"},
    ).json()["data"]
    assert hindi["intent"] == "candidate_alerts"
    assert hindi["scope"]["district_id"] == "OD-DIST-ganjam"
    assert hindi["scope"]["disease"] == "dengue"
    assert hindi["answer_state"] == "records_returned"

    english = client.post(
        "/api/v1/agent/query",
        json={"question": "How many dengue cases are there in Khordha?"},
    ).json()["data"]
    assert english["intent"] == "incidence_request"
    assert english["answer_state"] == "not_observable_from_public_sources"

    natural_future = client.post(
        "/api/v1/agent/query",
        json={"question": "Will dengue outbreak in Odisha in next 3 months?"},
    ).json()
    assert natural_future["data"]["intent"] == "forecast_request"
    assert natural_future["data"]["answer_state"] == "risk_factor_outlook"
    assert natural_future["data"]["evidence"] == []

    current_outlook = client.post(
        "/api/v1/agent/query",
        json={"question": "What is the current environmental outlook for dengue in Odisha?"},
    ).json()["data"]
    assert current_outlook["intent"] == "forecast_request"
    assert current_outlook["answer_state"] == "risk_factor_outlook"
    assert current_outlook["generation_mode"] == "policy_response"
    assert current_outlook["outlook"]["districts"]

    named_audit = client.post(
        "/api/v1/agent/query",
        json={"question": "Can the EpiClim historical data train an Odisha forecast?"},
    ).json()["data"]
    assert named_audit["intent"] == "data_audit"
    assert named_audit["answer_state"] == "audited_public_catalogue"
    assert "358 Odisha rows" in named_audit["answer"]
    assert "2 dengue rows" in named_audit["answer"]
    assert named_audit["reason_codes"] == [
        "POSITIVE_ONLY_CATALOGUE",
        "TARGET_SERIES_INELIGIBLE",
    ]
    assert named_audit["evidence"] == []

    evidence = client.post(
        "/api/v1/agent/query",
        json={"question": "Show dengue evidence for Khordha"},
    ).json()
    assert evidence["context"]["coverage_state"] == "fixture_fallback"
    assert all(item["is_fixture"] for item in evidence["data"]["evidence"])

    ambiguous = client.post(
        "/api/v1/agent/query",
        json={"question": "Compare dengue evidence in Ganjam and Khordha"},
    ).json()
    assert ambiguous["data"]["answer_state"] == "ambiguous_scope"
    assert ambiguous["data"]["reason_codes"] == ["AMBIGUOUS_QUERY_SCOPE"]
    assert set(ambiguous["data"]["scope_candidates"]["district_ids"]) == {
        "OD-DIST-ganjam",
        "OD-DIST-khordha",
    }
    assert ambiguous["data"]["evidence"] == []

    follow_up = client.post(
        "/api/v1/agent/query",
        json={
            "question": "What about the warning signals there?",
            "history": [
                {"role": "user", "content": "Show dengue evidence in Ganjam"},
                {"role": "assistant", "content": "Prior answer text is context, not evidence."},
            ],
        },
    )
    assert follow_up.status_code == 200
    follow_up_data = follow_up.json()["data"]
    assert follow_up_data["scope"]["district_id"] == "OD-DIST-ganjam"
    assert follow_up_data["scope"]["disease"] == "dengue"
    assert follow_up_data["scope"]["conversation_context_used"] is True

    for question in (
        "What treatment should I take for dengue?",
        "डेंगू के लिए कौन सी दवा और खुराक लूँ?",
        "ଡେଙ୍ଗୁ ପାଇଁ କେଉଁ ଔଷଧ ନେବି?",
    ):
        clinical = client.post("/api/v1/agent/query", json={"question": question}).json()
        assert clinical["data"]["intent"] == "clinical_advice_request"
        assert clinical["data"]["answer_state"] == "out_of_scope_clinical"
        assert clinical["data"]["reason_codes"] == ["CLINICAL_ADVICE_OUT_OF_SCOPE"]
        assert clinical["data"]["evidence"] == []


def test_official_statistics_questions_reach_the_bundled_annual_table() -> None:
    """Comparison, ranking and trend questions must not be refused as media searches.

    All of these were previously routed to the published-evidence path: the
    comparison was rejected as ambiguous because it named two districts, and the
    trend, burden and heatmap questions returned media record counts, while the
    official NCVBDC annual table holding every answer went unread. The
    evidence-vocabulary cases guard the inverse error -- answering a question
    about documents from a statistics table.
    """

    from services.api.evidence_agent import EvidenceAgent

    expectations = [
        (
            "Which district had more malaria cases in 2024, Koraput or Malkangiri?",
            "incidence_request",
        ),
        ("Has malaria in Odisha gone up or down since 2010?", "incidence_request"),
        ("Which district has the highest malaria burden?", "incidence_request"),
        # Ranking, mapping and distribution phrasings of the same question.
        ("Rank the districts by malaria burden", "incidence_request"),
        ("Show me the malaria heatmap across the state", "incidence_request"),
        ("Which districts report the most malaria cases?", "incidence_request"),
        ("What is the malaria pattern across the state?", "incidence_request"),
        # Year-to-year phrasings that never use a trend verb.
        ("How did malaria change between 2015 and 2024?", "incidence_request"),
        ("Malaria in Koraput from 2019 compared with 2024?", "incidence_request"),
        # Published records, not counted cases -- must stay on the evidence path.
        ("Compare dengue evidence in Ganjam and Khordha", "evidence_search"),
        ("Has dengue news coverage increased?", "evidence_search"),
        ("What dengue evidence exists for Khordha?", "evidence_search"),
        ("Which sources published the most dengue news?", "evidence_search"),
        ("Rank the districts by published dengue evidence", "evidence_search"),
    ]
    for question, expected in expectations:
        assert (
            EvidenceAgent._intent(question.casefold(), has_disease=True) == expected
        ), question
