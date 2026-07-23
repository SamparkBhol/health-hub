from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from services.api.main import create_app


def make_client(tmp_path) -> TestClient:
    return TestClient(create_app(f"sqlite:///{tmp_path / 'workflow.sqlite3'}"))


def test_fixture_replay_negation_and_append_only_review_correction(tmp_path) -> None:
    client = make_client(tmp_path)

    first = client.post("/api/v1/demo/replay-fixtures")
    assert first.status_code == 200
    assert first.json()["data"]["created_signals"] == 11
    assert first.json()["data"]["created_review_tasks"] == 9
    assert first.json()["data"]["created_catalogue_events"] == 1
    second = client.post("/api/v1/demo/replay-fixtures")
    assert second.json()["data"]["created_signals"] == 0
    assert second.json()["data"]["created_catalogue_events"] == 0

    signals = client.get("/api/v1/signals", params={"assertion": "all"}).json()["data"]
    assert len(signals) == 9
    assert all(item["processingState"] == "active_direct" for item in signals)
    assert any(item["assertion"] == "not_affirmed" for item in signals)
    assert all("98765 43210" not in item["evidence"] for item in signals)
    assert all(item["isFixture"] is True for item in signals)
    assert (
        client.get("/api/v1/signals").json()["context"]["coverage_state"]
        == "fixture_fallback"
    )
    tasks = client.get("/api/v1/review/tasks").json()["data"]
    assert len(tasks) == 9
    assert sum(item["task_kind"] == "quality_hold" for item in tasks) == 2
    assert all(item["assertion"] == "affirmed" for item in tasks)
    assert all(item["source_snapshot_id"] for item in tasks)
    assert all(item["snapshot_content_sha256"] for item in tasks)
    assert all(item["inspection_url"].startswith("fixture://bundled/") for item in tasks)
    assert all(item["retrieved_at"] for item in tasks)
    catalogue = client.get("/api/v1/layers/official_event_catalogue").json()["data"]
    assert len(catalogue) == 1
    assert catalogue[0]["event"]["positive_only_catalogue"] is True

    hold = next(item for item in tasks if item["task_kind"] == "quality_hold")
    hold_claim = client.post(
        f"/api/v1/review/tasks/{hold['id']}/claim",
        json={
            "reviewer_id": "privacy-reviewer",
            "expected_row_version": hold["row_version"],
        },
        headers={"Idempotency-Key": "claim-quality-hold"},
    ).json()["data"]["task"]
    invalid_promotion = client.post(
        f"/api/v1/review/tasks/{hold['id']}/decision",
        json={
            "reviewer_id": "privacy-reviewer",
            "expected_row_version": hold_claim["row_version"],
            "decision": "verified",
            "rationale": "Attempted invalid event promotion from a quality hold.",
            "event": {
                "district_id": hold["district_id"],
                "disease": hold["disease"],
            },
        },
        headers={"Idempotency-Key": "invalid-quality-hold-promotion"},
    )
    assert invalid_promotion.status_code == 409
    assert invalid_promotion.json()["data"]["code"] == "QUALITY_HOLD_NOT_EVENT_ELIGIBLE"
    assert client.get("/api/v1/layers/verified_event").json()["data"] == []

    task = next(item for item in tasks if item["task_kind"] == "event_verification")
    claim = client.post(
        f"/api/v1/review/tasks/{task['id']}/claim",
        json={"reviewer_id": "reviewer-a", "expected_row_version": task["row_version"]},
        headers={"Idempotency-Key": "claim-review-a", "If-Match": str(task["row_version"])},
    )
    assert claim.status_code == 200
    claimed = claim.json()["data"]["task"]
    assert claimed["state"] == "claimed"

    decided = client.post(
        f"/api/v1/review/tasks/{task['id']}/decision",
        json={
            "reviewer_id": "reviewer-a",
            "expected_row_version": claimed["row_version"],
            "decision": "verified",
            "rationale": "Source evidence is current and district-resolved.",
            "event": {"district_id": task["district_id"], "disease": task["disease"]},
        },
        headers={"Idempotency-Key": "decision-review-a", "If-Match": str(claimed["row_version"])},
    )
    assert decided.status_code == 200
    decision = decided.json()["data"]["decision"]
    events = client.get("/api/v1/layers/verified_event").json()["data"]
    assert len(events) == 1
    assert events[0]["decision_id"] == decision["id"]

    current_task = next(
        item
        for item in client.get("/api/v1/review/tasks").json()["data"]
        if item["id"] == task["id"]
    )
    correction = client.post(
        f"/api/v1/review/tasks/{task['id']}/decision",
        json={
            "reviewer_id": "adjudicator-b",
            "expected_row_version": current_task["row_version"],
            "decision": "rejected",
            "rationale": "Adjudication found the evidence was not an event.",
            "supersedes_decision_id": decision["id"],
        },
        headers={"Idempotency-Key": "correction-review-b"},
    )
    assert correction.status_code == 200
    assert correction.json()["data"]["decision"]["supersedes_id"] == decision["id"]
    # The old event remains in append-only storage but is no longer the active layer fact.
    assert client.get("/api/v1/layers/verified_event").json()["data"] == []


def test_job_idempotency_expired_lease_reclaim_and_stale_fence(tmp_path) -> None:
    client = make_client(tmp_path)
    request = {
        "source_id": "odisha_hfw_circulars_en",
        "kind": "parse",
        "payload_ref": "receipt:idsp-week-09",
        "payload_hash": "a" * 64,
    }
    enqueued = client.post(
        "/api/v1/internal/jobs/enqueue",
        json=request,
        headers={"Idempotency-Key": "enqueue-week-09"},
    )
    assert enqueued.status_code == 200
    job = enqueued.json()["data"]["job"]
    replay = client.post(
        "/api/v1/internal/jobs/enqueue",
        json=request,
        headers={"Idempotency-Key": "enqueue-week-09"},
    )
    assert replay.json()["data"]["idempotent_replay"] is True
    assert replay.json()["data"]["job"]["id"] == job["id"]

    first_claim = client.post(
        "/api/v1/internal/jobs/claim", json={"owner": "worker-old", "lease_seconds": 15}
    ).json()["data"]["job"]
    assert first_claim["fencing_token"] == 1

    database = client.app.state.database
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    with database.transaction() as connection:
        connection.execute("UPDATE job SET lease_expires_at=? WHERE id=?", (expired, job["id"]))

    second_claim = client.post(
        "/api/v1/internal/jobs/claim", json={"owner": "worker-new", "lease_seconds": 60}
    ).json()["data"]["job"]
    assert second_claim["id"] == job["id"]
    assert second_claim["fencing_token"] == 2

    signal = {
        "source_id": "odisha_hfw_circulars_en",
        "source_snapshot_id": "job:0",
        "district_id": "OD-DIST-khordha",
        "disease": "dengue",
        "assertion": "affirmed",
        "evidence_text": "Synthetic fixture: dengue evidence in Khordha.",
        "evidence_start": 0,
        "evidence_end": 46,
        "content_sha256": "b" * 64,
        "retrieved_at": "2026-07-21T00:00:00Z",
    }
    stale = client.post(
        f"/api/v1/internal/jobs/{job['id']}/complete",
        json={"owner": "worker-old", "fencing_token": 1, "signals": [signal]},
        headers={"Idempotency-Key": "complete-old"},
    )
    assert stale.status_code == 409
    assert stale.json()["data"]["code"] == "STALE_FENCING_TOKEN"

    completed = client.post(
        f"/api/v1/internal/jobs/{job['id']}/complete",
        json={"owner": "worker-new", "fencing_token": 2, "signals": [signal]},
        headers={"Idempotency-Key": "complete-new"},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["data"]["job"]["state"] == "completed"
    assert len(completed.json()["data"]["review_task_ids"]) == 1
    replayed = client.post(
        f"/api/v1/internal/jobs/{job['id']}/complete",
        json={"owner": "worker-new", "fencing_token": 2, "signals": [signal]},
        headers={"Idempotency-Key": "complete-new"},
    )
    assert replayed.status_code == 200
    assert replayed.json()["data"]["idempotent_replay"] is True
