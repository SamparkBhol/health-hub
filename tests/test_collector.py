from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from packages.contracts.api import (
    LIVE_EVIDENCE_PLACEHOLDER,
    LIVE_EVIDENCE_REDACTION_STATE,
    RedactedSignalInput,
    SourceReceiptInput,
)
from services.api import collection_runtime as collection_runtime_module
from services.api.collection_runtime import CollectionRuntime, _signal_inputs
from services.api.database import Database
from workers.ingestion.connectors import DiscoveredLink, IngestionOutcome
from workers.ingestion.idsp import IdspCatalogueRow
from workers.ingestion.models import Document, FetchReceipt
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.registry import load_registry
from workers.ingestion.robots import RobotsVerdict
from workers.ingestion.safe_fetch import (
    DEFAULT_CRAWLER_CONTACT,
    FetchPolicy,
    crawler_contact,
)


def test_collector_identifies_itself_without_an_operator_contact(monkeypatch) -> None:
    """An unset (or placeholder) CRAWLER_CONTACT must not disable collection.

    Withholding every request until an operator exported a mailbox is what made
    the shipped product a fixture demo.  The courtesy an origin is actually owed
    is a self-identifying User-Agent, and one is always present.
    """

    monkeypatch.delenv("CRAWLER_CONTACT", raising=False)
    assert crawler_contact() == DEFAULT_CRAWLER_CONTACT
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:replace-with-monitored@example.invalid")
    assert crawler_contact() == DEFAULT_CRAWLER_CONTACT
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:ops@health.example.org")
    assert crawler_contact() == "mailto:ops@health.example.org"
    policy = FetchPolicy.load()
    assert policy.user_agent.startswith("OdishaPublicHealthEvidenceBot/")
    assert "mailto:ops@health.example.org" in policy.user_agent


def test_collector_withholds_only_when_live_collection_is_switched_off(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("CRAWLER_CONTACT", raising=False)
    monkeypatch.setenv("LIVE_COLLECTION_ENABLED", "false")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    runtime = CollectionRuntime(database)
    assert runtime.tick(maximum_jobs=3) == {
        "state": "withheld",
        "reason_code": "LIVE_COLLECTION_DISABLED",
        "enqueued": 0,
        "processed": [],
    }
    runtime.start()
    assert runtime.scheduler is None


def test_in_process_scheduler_and_live_collection_are_on_by_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("ENABLE_IN_PROCESS_SCHEDULER", raising=False)
    monkeypatch.delenv("LIVE_COLLECTION_ENABLED", raising=False)
    runtime = CollectionRuntime(Database(f"sqlite:///{tmp_path / 'defaults.sqlite3'}"))
    assert runtime.enabled is True
    assert runtime.live_collection_enabled is True
    assert runtime.contact_configured is True
    status = runtime.status()
    assert status["api_jobs_per_tick"] > 1
    assert status["fetch_workers"] > 1
    assert status["enabled_routes"] > 7
    assert set(status["route_languages"]) >= {"or", "hi", "en"}


def test_collector_claims_registered_indices_before_detail_backlog(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:security@example.org")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    database.enqueue_job(
        source_id="odisha_hfw_circulars_en",
        kind="fetch",
        payload_ref="registered-link:https://health.odisha.gov.in/reports/queued.pdf",
        payload_hash="a" * 64,
        idempotency_key="preexisting-detail-backlog",
    )
    runtime = CollectionRuntime(database)
    seen: list[dict] = []

    def record(job: dict) -> dict:
        seen.append(job)
        return {"job_id": job["id"], "payload_ref": job["payload_ref"]}

    monkeypatch.setattr(runtime, "_process", record)
    result = runtime.tick(maximum_jobs=1)
    assert result["processed"]
    assert seen[0]["kind"] == "discover"
    assert str(seen[0]["payload_ref"]).startswith("registered-index:")


class _AllowAllRobots:
    """Offline stand-in for RobotsPolicy in tests that stub the network seam."""

    def evaluate(self, url: str) -> RobotsVerdict:  # noqa: ARG002 - fixed verdict
        return RobotsVerdict(allowed=True, state="allowed", crawl_delay=0.0)


def _offline(runtime: CollectionRuntime) -> CollectionRuntime:
    runtime.robots = _AllowAllRobots()  # type: ignore[assignment]
    return runtime


def test_collector_runs_allowlisted_outcome_through_snapshot_and_review(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:security@example.org")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    pipeline = IngestionPipeline.default()

    def fake_ingest(*, source_id: str, url: str, **kwargs) -> IngestionOutcome:
        is_detail = url.endswith("/health-detail")
        text = "Synthetic network seam: dengue evidence in Khordha."
        digest = hashlib.sha256(text.encode()).hexdigest()
        retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
        document = Document(
            document_id=f"doc_{digest[:20]}",
            source_id=source_id,
            canonical_url=url,
            retrieved_at=retrieved,
            content_type="text/html",
            text=text,
            sha256=digest,
        )
        return IngestionOutcome(
            receipt=FetchReceipt(
                source_id=source_id,
                requested_url=url,
                final_url=url,
                retrieved_at=retrieved,
                status_code=200,
                content_type="text/html",
                byte_length=len(text.encode()),
                sha256=digest,
            ),
            signal=pipeline.process(document) if is_detail else None,
            discovered_links=(
                ()
                if is_detail
                else (
                    DiscoveredLink(
                        url="https://health.odisha.gov.in/health-detail",
                        label="dengue health bulletin",
                        content_hint="text/html",
                    ),
                )
            ),
        )

    monkeypatch.setattr(
        "services.api.collection_runtime.ingest_registered_url", fake_ingest
    )
    runtime = _offline(CollectionRuntime(database))
    discovery = runtime.tick(maximum_jobs=1)
    assert discovery["state"] == "completed"
    assert discovery["enqueued"] >= 5
    assert discovery["processed"][0]["state"] == "completed"
    assert discovery["processed"][0]["signal_count"] == 0
    assert discovery["processed"][0]["discovered_jobs"] == 1

    # Discovery indices are deliberately drained before detail links, and the
    # registry now holds a hundred-plus routes, so the detail job is reached by
    # widening the per-tick budget rather than by ticking one job at a time.
    result = None
    for _ in range(6):
        candidate = runtime.tick(maximum_jobs=200)
        matched = [
            item for item in candidate["processed"] if item.get("signal_count") == 1
        ]
        if matched:
            result = matched[0]
            break
    assert result is not None
    assert result["state"] == "completed"
    signals = database.list_signals()
    assert len(signals) == 1  # noqa: PLR2004
    assert signals[0]["district_id"] == "OD-DIST-khordha"
    assert signals[0]["disease"] == "dengue"
    assert signals[0]["access_path"] == "live_origin"
    assert len(database.list_review_tasks()) == 1


def _complete_next_link(database: Database, owner: str, sequence: int) -> None:
    job = database.claim_job(
        owner=owner,
        lease_seconds=300,
        kind="fetch",
        payload_prefix="registered-link:",
    )
    assert job is not None
    payload_url = str(job["payload_ref"]).removeprefix("registered-link:")
    sha256 = hashlib.sha256(f"body-{sequence}".encode()).hexdigest()
    database.complete_job(
        job_id=job["id"],
        owner=owner,
        fencing_token=job["fencing_token"],
        idempotency_key=f"complete-link-{sequence}",
        receipt=SourceReceiptInput(
            source_snapshot_id=f"snapshot_link_{sequence}",
            source_id=job["source_id"],
            requested_url=payload_url,
            final_url=payload_url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="text/html",
            byte_length=10,
            sha256=sha256,
            access_path="live_origin",
        ),
        signals=[],
    )


def test_discovered_link_queue_progresses_beyond_first_two_across_restarts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:security@example.org")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    source_id = "odisha_hfw_circulars_en"
    links = tuple(
        DiscoveredLink(
            url=f"https://health.odisha.gov.in/reports/{number}.pdf",
            label=f"health bulletin {number}",
            content_hint="application/pdf",
        )
        for number in range(5)
    )
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source_id,
            requested_url="https://health.odisha.gov.in/en/notifications/circulars",
            final_url="https://health.odisha.gov.in/en/notifications/circulars",
            retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            status_code=200,
            content_type="text/html",
            byte_length=100,
            sha256="a" * 64,
        ),
        signal=None,
        discovered_links=links,
    )
    source = CollectionRuntime(database).registry.get(source_id)
    # The shipped window is much wider now (see the default-window test below);
    # narrowing it here keeps this test about *resuming* a partly drained queue.
    monkeypatch.setattr(collection_runtime_module, "MAX_DISCOVERED_LINKS", 2)

    first_runtime = CollectionRuntime(database)
    assert first_runtime._enqueue_discovered(source, outcome) == 2
    first_rows = database.list_discovered_links(source_id)
    assert [row["state"] for row in first_rows].count("queued") == 2
    assert [row["state"] for row in first_rows].count("pending") == 3
    assert all(len(row["label_sha256"]) == 64 for row in first_rows)
    assert all("health bulletin" not in str(row) for row in first_rows)

    _complete_next_link(database, "worker-a", 1)
    _complete_next_link(database, "worker-a", 2)
    restarted = CollectionRuntime(database)
    assert restarted._enqueue_discovered(source, outcome) == 2
    _complete_next_link(database, "worker-b", 3)
    _complete_next_link(database, "worker-b", 4)
    assert CollectionRuntime(database)._enqueue_discovered(source, outcome) == 1
    _complete_next_link(database, "worker-c", 5)

    final_rows = database.list_discovered_links(source_id)
    assert len(final_rows) == 5
    assert {row["state"] for row in final_rows} == {"fetched"}
    assert len({row["job_id"] for row in final_rows}) == 5


def test_html_only_source_does_not_queue_unrelated_pdf_links(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    runtime = CollectionRuntime(database)
    source = runtime.registry.get("ganjam_collectorate")
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source.id,
            requested_url=source.url,
            final_url=source.url,
            retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            status_code=200,
            content_type="text/html",
            byte_length=100,
            sha256="e" * 64,
        ),
        signal=None,
        discovered_links=(
            DiscoveredLink(
                url="https://ganjam.odisha.gov.in/site-policy.pdf",
                label="Site policy",
                content_hint="application/pdf",
            ),
            DiscoveredLink(
                url="https://ganjam.odisha.gov.in/dengue-update",
                label="Dengue health update",
                content_hint="text/html",
            ),
        ),
    )
    assert runtime._enqueue_discovered(source, outcome) == 1
    stored = database.list_discovered_links(source.id)
    assert [item["url"] for item in stored] == [
        "https://ganjam.odisha.gov.in/dengue-update"
    ]


def test_section_route_rejects_navigation_pages_even_when_label_says_health(
    tmp_path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'navigation.sqlite3'}")
    runtime = CollectionRuntime(database)
    source = runtime.registry.get("sambad_district_bhadrak_or")
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source.id,
            requested_url=source.url,
            final_url=source.url,
            retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            status_code=200,
            content_type="text/html",
            byte_length=100,
            sha256="1" * 64,
        ),
        signal=None,
        discovered_links=(
            DiscoveredLink(
                url="https://sambad.in/district",
                label="Dengue health news by district",
                content_hint="text/html",
                score=8,
            ),
            DiscoveredLink(
                url="https://sambad.in/crime",
                label="Health and hospital crime news",
                content_hint="text/html",
                score=3,
            ),
            DiscoveredLink(
                url="https://sambad.in/odisha/bhadrak-dengue-update-2026",
                label="Bhadrak dengue update",
                content_hint="text/html",
                score=5,
            ),
        ),
    )
    assert runtime._enqueue_discovered(source, outcome) == 1
    assert [row["url"] for row in database.list_discovered_links()] == [
        "https://sambad.in/odisha/bhadrak-dengue-update-2026"
    ]


def test_discovered_url_has_one_global_owner_and_tracking_variants_collapse(
    tmp_path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'global-url.sqlite3'}")
    runtime = CollectionRuntime(database)
    first = runtime.registry.get("sambad_district_bhadrak_or")
    second = runtime.registry.get("sambad_district_cuttack_or")

    def outcome(source_id: str, url: str) -> IngestionOutcome:
        return IngestionOutcome(
            receipt=FetchReceipt(
                source_id=source_id,
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                status_code=200,
                content_type="text/html",
                byte_length=100,
                sha256="2" * 64,
            ),
            signal=None,
            discovered_links=(
                DiscoveredLink(
                    url=url,
                    label="Dengue update in Odisha",
                    content_hint="text/html",
                    score=5,
                ),
            ),
        )

    base = "https://sambad.in/odisha/dengue-update?id=42"
    assert runtime._enqueue_discovered(
        first, outcome(first.id, f"{base}&utm_source=bhadrak#top")
    ) == 1
    assert runtime._enqueue_discovered(
        second, outcome(second.id, f"{base}&utm_source=cuttack")
    ) == 0
    rows = database.list_discovered_links()
    assert len(rows) == 1
    assert rows[0]["source_id"] == first.id
    assert rows[0]["url"] == base


def test_startup_migrates_preexisting_cross_route_url_duplicates(tmp_path) -> None:
    path = tmp_path / "legacy-global-url.sqlite3"
    database = Database(f"sqlite:///{path}")
    first = "sambad_district_bhadrak_or"
    second = "sambad_district_cuttack_or"
    urls = (
        "https://sambad.in/odisha/dengue-update?id=42&utm_source=bhadrak",
        "https://sambad.in/odisha/dengue-update?utm_source=cuttack&id=42#top",
    )
    with database.transaction() as connection:
        connection.execute("DROP INDEX idx_discovered_url_global")
        for source_id, url in zip((first, second), urls, strict=True):
            connection.execute(
                """
                INSERT INTO discovered_link(
                  source_id,url,url_sha256,label_sha256,content_hint,priority_rank,
                  state,first_seen_at,last_seen_at
                ) VALUES(?,?,?,?,?,0,'pending',?,?)
                """,
                (
                    source_id,
                    url,
                    hashlib.sha256(url.encode()).hexdigest(),
                    "4" * 64,
                    "text/html",
                    "2026-07-21T14:00:00Z",
                    "2026-07-21T14:00:00Z",
                ),
            )
    database.close()

    migrated = Database(f"sqlite:///{path}")
    rows = migrated.list_discovered_links()
    assert len(rows) == 1
    assert rows[0]["url"] == "https://sambad.in/odisha/dengue-update?id=42"


def test_unapproved_pdf_waits_for_exact_hash_then_gets_new_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:security@example.org")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    source_id = "odisha_hfw_circulars_en"
    url = "https://health.odisha.gov.in/reports/scanned.pdf"
    source = CollectionRuntime(database).registry.get(source_id)
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source_id,
            requested_url=source.url,
            final_url=source.url,
            retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            status_code=200,
            content_type="text/html",
            byte_length=50,
            sha256="b" * 64,
        ),
        signal=None,
        discovered_links=(
            DiscoveredLink(url=url, label="health PDF", content_hint="application/pdf"),
        ),
    )
    runtime = CollectionRuntime(database)
    assert runtime._enqueue_discovered(source, outcome) == 1
    job = database.claim_job(
        owner="metadata-worker",
        lease_seconds=300,
        kind="fetch",
        payload_prefix="registered-link:",
    )
    assert job is not None
    observed_sha256 = "c" * 64
    database.complete_job(
        job_id=job["id"],
        owner="metadata-worker",
        fencing_token=job["fencing_token"],
        idempotency_key="metadata-complete",
        receipt=SourceReceiptInput(
            source_snapshot_id="snapshot_unapproved_pdf",
            source_id=source_id,
            requested_url=url,
            final_url=url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="application/pdf",
            byte_length=100,
            sha256=observed_sha256,
            access_path="live_origin",
        ),
        signals=[],
        link_disposition="pending_approval",
    )
    waiting = database.list_discovered_links(source_id)[0]
    assert waiting["state"] == "pending_approval"
    assert waiting["observed_content_sha256"] == observed_sha256
    assert runtime._enqueue_pending_links(source) == 0

    monkeypatch.setenv("APPROVED_PDF_SHA256S", observed_sha256)
    approved_runtime = CollectionRuntime(database)
    assert approved_runtime._enqueue_pending_links(source) == 1
    approved = database.list_discovered_links(source_id)[0]
    assert approved["state"] == "queued"
    assert approved["queue_mode"] == "approved"
    assert approved["job_id"] != job["id"]


def test_approved_idsp_report_persists_positive_only_catalogue_rows(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CRAWLER_CONTACT", "mailto:security@example.org")
    database = Database(f"sqlite:///{tmp_path / 'collector.sqlite3'}")
    url = "https://idsp.mohfw.gov.in/WriteReadData/l892s/synthetic.pdf"
    digest = hashlib.sha256(b"synthetic idsp").hexdigest()
    database.enqueue_job(
        source_id="idsp_weekly_outbreaks",
        kind="fetch",
        payload_ref=f"registered-link:{url}",
        payload_hash=hashlib.sha256(url.encode()).hexdigest(),
        idempotency_key="test-idsp-detail",
    )

    def fake_ingest(**kwargs) -> IngestionOutcome:
        return IngestionOutcome(
            receipt=FetchReceipt(
                source_id=kwargs["source_id"],
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                status_code=200,
                content_type="application/pdf",
                byte_length=100,
                sha256=digest,
            ),
            signal=None,
            catalogue_rows=(
                IdspCatalogueRow(
                    outbreak_id="OR/ANU/2026/9/334",
                    year=2026,
                    week=9,
                    district_code="ANU",
                    source_text="OR/ANU/2026/9/334 dengue in Angul",
                ),
            ),
            processing_state="positive_only_official_catalogue",
        )

    monkeypatch.setattr(
        "services.api.collection_runtime.ingest_registered_url", fake_ingest
    )
    result = CollectionRuntime(database).tick(maximum_jobs=1)
    assert result["processed"][0]["catalogue_event_count"] == 1
    event = database.list_catalogue_events()[0]
    assert event["district_id"] == "OD-DIST-angul"
    assert event["disease"] == "dengue"
    assert event["event"]["positive_only_catalogue"] is True
    assert event["event"]["missing_weeks_are_not_zero"] is True


def test_live_document_uses_fixed_non_retained_evidence_contract() -> None:
    pipeline = IngestionPipeline.default()
    text = ("background material. " * 900) + "Dengue reported in Cuttack."
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        document_id="long-document",
        source_id="odisha_hfw_circulars_en",
        canonical_url="https://health.odisha.gov.in/long",
        retrieved_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        content_type="text/html",
        text=text,
        sha256=digest,
    )
    extracted = pipeline.process(document)
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id="odisha_hfw_circulars_en",
            requested_url=document.canonical_url,
            final_url=document.canonical_url,
            retrieved_at=document.retrieved_at,
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
        ),
        signal=extracted,
    )
    receipt = SourceReceiptInput(
        source_snapshot_id="snapshot_long",
        source_id="odisha_hfw_circulars_en",
        requested_url=document.canonical_url,
        final_url=document.canonical_url,
        retrieved_at="2026-07-21T14:00:00Z",
        status_code=200,
        content_type="text/html",
        byte_length=len(text.encode()),
        sha256=digest,
        access_path="live_origin",
    )
    signals = _signal_inputs(outcome, receipt, pipeline)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.evidence_text == LIVE_EVIDENCE_PLACEHOLDER
    assert signal.evidence_start == 0
    assert signal.evidence_end - signal.evidence_start == len(signal.evidence_text)
    assert "Dengue reported in Cuttack" not in signal.evidence_text
    assert signal.redaction_state == LIVE_EVIDENCE_REDACTION_STATE
    assert signal.content_sha256 == hashlib.sha256(
        LIVE_EVIDENCE_PLACEHOLDER.encode()
    ).hexdigest()
    Database._validate_signal(signal, "odisha_hfw_circulars_en")


@pytest.mark.parametrize(
    ("language", "text", "forbidden", "heuristic_missed"),
    (
        (
            "en",
            "This is live source material. Dengue cases were reported in Khordha "
            "district. Patient Aroop Mishra remains under care.",
            "Aroop Mishra",
            True,
        ),
        (
            "hi",
            "यह लाइव स्रोत सामग्री है। गंजाम जिले में डेंगू का मामला दर्ज किया गया। "
            "रोगी राकेश नायक अस्पताल में है।",
            "राकेश नायक",
            True,
        ),
        (
            "or",
            "ଏହା ଲାଇଭ୍ ଉତ୍ସ ସାମଗ୍ରୀ। ଖୋର୍ଦ୍ଧା ଜିଲ୍ଲାରେ ଡେଙ୍ଗୁ ମାମଲା ଚିହ୍ନଟ "
            "ହୋଇଛି। ରୋଗୀ ରମେଶ ନାୟକ ଚିକିତ୍ସାଧୀନ।",
            "ରମେଶ ନାୟକ",
            True,
        ),
        (
            "en",
            "Dengue cases were reported in Khordha district. Call 9876543210.",
            "9876543210",
            False,
        ),
        (
            "en",
            "Dengue cases were reported in Khordha district. "
            "Email rahul@example.org.",
            "rahul@example.org",
            False,
        ),
    ),
)
def test_live_names_and_contacts_never_reach_persisted_signals(
    tmp_path,
    language: str,
    text: str,
    forbidden: str,
    heuristic_missed: bool,
) -> None:
    source_id = "odisha_hfw_circulars_en"
    source_url = f"https://health.odisha.gov.in/privacy-boundary/{language}"
    retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
    digest = hashlib.sha256(text.encode()).hexdigest()
    pipeline = IngestionPipeline.default()
    document = Document(
        document_id=f"privacy-{language}-{digest[:8]}",
        source_id=source_id,
        canonical_url=source_url,
        retrieved_at=retrieved,
        content_type="text/html",
        text=text,
        sha256=digest,
    )
    extracted = pipeline.process(document)
    assert extracted.diseases == ("dengue",)
    assert extracted.districts
    assert (forbidden in extracted.redacted_evidence) is heuristic_missed
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source_id,
            requested_url=source_url,
            final_url=source_url,
            retrieved_at=retrieved,
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
        ),
        signal=extracted,
    )
    receipt = SourceReceiptInput(
        source_snapshot_id=f"snapshot_privacy_{digest[:24]}",
        source_id=source_id,
        requested_url=source_url,
        final_url=source_url,
        retrieved_at="2026-07-21T14:00:00Z",
        status_code=200,
        content_type="text/html",
        byte_length=len(text.encode()),
        sha256=digest,
        access_path="live_origin",
    )
    signals = _signal_inputs(outcome, receipt, pipeline)
    assert signals[0].disease == "dengue"
    assert signals[0].district_id is not None

    database = Database(f"sqlite:///{tmp_path / 'privacy.sqlite3'}")
    job, _ = database.enqueue_job(
        source_id=source_id,
        kind="fetch",
        payload_ref=f"registered-link:{source_url}",
        payload_hash=hashlib.sha256(source_url.encode()).hexdigest(),
        idempotency_key=f"privacy-{digest}",
    )
    claimed = database.claim_job(
        owner="privacy-test", lease_seconds=300, job_id=job["id"]
    )
    assert claimed is not None
    database.complete_job(
        job_id=job["id"],
        owner="privacy-test",
        fencing_token=claimed["fencing_token"],
        idempotency_key=f"complete-{digest}",
        receipt=receipt,
        signals=signals,
    )

    persisted = database.list_signals(fixture_mode="live_only")
    assert len(persisted) == 1
    assert persisted[0]["evidence_text"] == LIVE_EVIDENCE_PLACEHOLDER
    assert persisted[0]["redaction_state"] == LIVE_EVIDENCE_REDACTION_STATE
    assert persisted[0]["evidence_start"] == 0
    assert persisted[0]["evidence_end"] == len(LIVE_EVIDENCE_PLACEHOLDER)
    assert persisted[0]["content_sha256"] == hashlib.sha256(
        LIVE_EVIDENCE_PLACEHOLDER.encode()
    ).hexdigest()
    tasks = database.list_review_tasks()
    assert tasks[0]["evidence_text"] == LIVE_EVIDENCE_PLACEHOLDER
    assert tasks[0]["registered_source_url"].startswith(
        "https://health.odisha.gov.in/"
    )
    durable_views = json.dumps(
        {"signals": persisted, "tasks": tasks}, ensure_ascii=False
    )
    assert forbidden not in durable_views


def test_startup_migrates_legacy_live_evidence_and_evidence_hashes(tmp_path) -> None:
    database_path = tmp_path / "legacy-live.sqlite3"
    database = Database(f"sqlite:///{database_path}")
    source_id = "odisha_hfw_circulars_en"
    source_url = "https://health.odisha.gov.in/legacy-live"
    job, _ = database.enqueue_job(
        source_id=source_id,
        kind="fetch",
        payload_ref=f"registered-link:{source_url}",
        payload_hash="a" * 64,
        idempotency_key="legacy-live-enqueue",
    )
    claimed = database.claim_job(
        owner="legacy-live-test", lease_seconds=300, job_id=job["id"]
    )
    assert claimed is not None
    original = "Dengue in Khordha concerns patient Aroop Mishra."
    database.complete_job(
        job_id=job["id"],
        owner="legacy-live-test",
        fencing_token=claimed["fencing_token"],
        idempotency_key="legacy-live-complete",
        receipt=SourceReceiptInput(
            source_snapshot_id="snapshot_legacy_live",
            source_id=source_id,
            requested_url=source_url,
            final_url=source_url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="text/html",
            byte_length=len(original.encode()),
            sha256="b" * 64,
            access_path="live_origin",
        ),
        signals=[
            RedactedSignalInput(
                source_id=source_id,
                source_snapshot_id="snapshot_legacy_live",
                district_id="OD-DIST-khordha",
                disease="dengue",
                assertion="affirmed",
                evidence_text=original,
                evidence_start=0,
                evidence_end=len(original),
                content_sha256=hashlib.sha256(original.encode()).hexdigest(),
                retrieved_at="2026-07-21T14:00:00Z",
                processing_state="active_direct",
                language="en",
            )
        ],
    )
    legacy_hash = hashlib.sha256(original.encode()).hexdigest()
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE signal SET content_key=?, evidence_text=?, evidence_start=0,
                evidence_end=?, content_sha256=?, redaction_state='heuristic_unvalidated'
            """,
            (legacy_hash, original, len(original), legacy_hash),
        )
    database.close()

    migrated = Database(f"sqlite:///{database_path}")
    row = migrated.list_signals(fixture_mode="live_only")[0]
    assert row["evidence_text"] == LIVE_EVIDENCE_PLACEHOLDER
    assert row["content_sha256"] == hashlib.sha256(
        LIVE_EVIDENCE_PLACEHOLDER.encode()
    ).hexdigest()
    assert row["content_key"] != legacy_hash
    assert row["redaction_state"] == LIVE_EVIDENCE_REDACTION_STATE
    assert "Aroop Mishra" not in json.dumps(row, ensure_ascii=False)


def test_one_district_document_yields_one_signal_per_disease(tmp_path) -> None:
    """A page fixed to one district is not entity-linkage ambiguity.

    A district health-department page names malaria, leprosy and tuberculosis in
    the same paragraph. The old rule required exactly one disease and threw the
    whole page away, which is why the district map stayed empty.
    """

    pipeline = IngestionPipeline.default()
    text = (
        "Khordha district health department controls malaria, leprosy and "
        "tuberculosis through the district programme."
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
    document = Document(
        document_id=f"doc_{digest[:20]}",
        source_id="district_khordha_health_en",
        canonical_url="https://khordha.odisha.gov.in/en/departments/health",
        retrieved_at=retrieved,
        content_type="text/html",
        text=text,
        sha256=digest,
    )
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=document.source_id,
            requested_url=document.canonical_url,
            final_url=document.canonical_url,
            retrieved_at=retrieved,
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
        ),
        signal=pipeline.process(document),
    )
    receipt = SourceReceiptInput(
        source_snapshot_id="snapshot_district_health",
        source_id=document.source_id,
        requested_url=document.canonical_url,
        final_url=document.canonical_url,
        retrieved_at="2026-07-21T14:00:00Z",
        status_code=200,
        content_type="text/html",
        byte_length=len(text.encode()),
        sha256=digest,
        access_path="live_origin",
    )
    source = load_registry().get(document.source_id)
    signals = _signal_inputs(outcome, receipt, pipeline, source)
    assert {signal.disease for signal in signals} == {
        "malaria",
        "leprosy",
        "tuberculosis",
    }
    assert {signal.district_id for signal in signals} == {"OD-DIST-khordha"}
    assert len({signal.signal_id for signal in signals}) == len(signals)
    for signal in signals:
        Database._validate_signal(signal, document.source_id)

    # The Cartesian rule still holds: on a route that is not published by any
    # one district, two districts plus two diseases stays one unlocated,
    # unclassified row.
    statewide = load_registry().get("odisha_hfw_circulars_en")
    assert statewide.district_id is None
    ambiguous_text = (
        "Dengue was reported in Khordha district and cholera in Ganjam district."
    )
    ambiguous_digest = hashlib.sha256(ambiguous_text.encode()).hexdigest()
    ambiguous = IngestionOutcome(
        receipt=outcome.receipt,
        signal=pipeline.process(
            Document(
                document_id=f"doc_{ambiguous_digest[:20]}",
                source_id=statewide.id,
                canonical_url=statewide.url,
                retrieved_at=retrieved,
                content_type="text/html",
                text=ambiguous_text,
                sha256=ambiguous_digest,
            )
        ),
    )
    statewide_receipt = receipt.model_copy(update={"source_id": statewide.id})
    rows = _signal_inputs(ambiguous, statewide_receipt, pipeline, statewide)
    assert len(rows) == 1
    assert rows[0].disease is None
    assert rows[0].district_id is None
    assert rows[0].processing_state == "ambiguous_entity_linkage"


def test_sentence_linkage_does_not_cross_assign_a_multi_item_document() -> None:
    pipeline = IngestionPipeline.default()
    retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
    text = (
        "Dengue cases were reported in Gajapati district. "
        "The national malaria programme issued guidance. "
        "Tuberculosis and cholera were discussed in other state reports."
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = Document(
        document_id=f"doc_{digest[:20]}",
        source_id="sambad_district_gajapati_or",
        canonical_url="https://sambad.in/odisha/multi-item-report",
        retrieved_at=retrieved,
        content_type="text/html",
        text=text,
        sha256=digest,
    )
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=document.source_id,
            requested_url=document.canonical_url,
            final_url=document.canonical_url,
            retrieved_at=retrieved,
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
        ),
        signal=pipeline.process(document),
    )
    receipt = SourceReceiptInput(
        source_snapshot_id="snapshot_multi_item",
        source_id=document.source_id,
        requested_url=document.canonical_url,
        final_url=document.canonical_url,
        retrieved_at="2026-07-21T14:00:00Z",
        status_code=200,
        content_type="text/html",
        byte_length=len(text.encode()),
        sha256=digest,
        access_path="live_origin",
    )
    source = load_registry().get(document.source_id)
    rows = _signal_inputs(outcome, receipt, pipeline, source)
    active = [row for row in rows if row.processing_state == "active_direct"]
    held = [row for row in rows if row.processing_state == "ambiguous_entity_linkage"]
    assert [(row.disease, row.district_id) for row in active] == [
        ("dengue", "OD-DIST-gajapati")
    ]
    assert len(held) == 1
    assert held[0].district_id is None


def test_unchanged_document_creates_two_snapshots_but_one_logical_signal(
    tmp_path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'logical-dedup.sqlite3'}")
    source_id = "odisha_hfw_circulars_en"
    url = "https://health.odisha.gov.in/en/dengue-bulletin"
    document_sha = "3" * 64
    for sequence in (1, 2):
        job, _ = database.enqueue_job(
            source_id=source_id,
            kind="fetch",
            payload_ref=f"registered-index:{source_id}",
            payload_hash=document_sha,
            idempotency_key=f"logical-dedup-job-{sequence}",
        )
        claimed = database.claim_job(
            owner="dedup-worker", lease_seconds=300, job_id=job["id"]
        )
        assert claimed is not None
        retrieved_at = f"2026-07-{20 + sequence:02d}T14:00:00Z"
        snapshot_id = f"snapshot_logical_{sequence}"
        database.complete_job(
            job_id=job["id"],
            owner="dedup-worker",
            fencing_token=claimed["fencing_token"],
            idempotency_key=f"logical-dedup-complete-{sequence}",
            receipt=SourceReceiptInput(
                source_snapshot_id=snapshot_id,
                source_id=source_id,
                requested_url=url,
                final_url=url,
                retrieved_at=retrieved_at,
                status_code=200,
                content_type="text/html",
                byte_length=100,
                sha256=document_sha,
                access_path="live_origin",
            ),
            signals=[
                RedactedSignalInput(
                    source_id=source_id,
                    source_snapshot_id=snapshot_id,
                    district_id="OD-DIST-khordha",
                    disease="dengue",
                    assertion="affirmed",
                    evidence_text="Dengue in Khordha.",
                    evidence_start=0,
                    evidence_end=len("Dengue in Khordha."),
                    content_sha256=hashlib.sha256(
                        b"Dengue in Khordha."
                    ).hexdigest(),
                    retrieved_at=retrieved_at,
                    processing_state="active_direct",
                    language="en",
                    extractor_version="rules-sentence-link-v2",
                )
            ],
        )
    assert len(database.list_signals(fixture_mode="live_only")) == 1
    with database.transaction() as connection:
        snapshot_count = connection.execute(
            "SELECT COUNT(*) AS total FROM source_snapshot WHERE source_id=?",
            (source_id,),
        ).fetchone()["total"]
    assert snapshot_count == 2


def test_publisher_district_wins_over_an_incidental_second_district(tmp_path) -> None:
    """A district's own page that also prints a Bhubaneswar address is still its own.

    The allowance is narrow: it applies only to a registered `document` route
    whose own district is among the districts the text resolves, and never to a
    followed detail link.
    """

    pipeline = IngestionPipeline.default()
    source = load_registry().get("district_bhadrak_health_en")
    assert source.district_id == "OD-DIST-bhadrak"
    assert source.content_role == "document"
    text = (
        "Bhadrak district health department, malaria control cell. "
        "Correspondence: Directorate of Health Services, Bhubaneswar."
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
    document = Document(
        document_id=f"doc_{digest[:20]}",
        source_id=source.id,
        canonical_url=source.url,
        retrieved_at=retrieved,
        content_type="text/html",
        text=text,
        sha256=digest,
    )
    signal = pipeline.process(document)
    assert {match.district_id for match in signal.districts} == {
        "OD-DIST-bhadrak",
        "OD-DIST-khordha",
    }
    outcome = IngestionOutcome(
        receipt=FetchReceipt(
            source_id=source.id,
            requested_url=source.url,
            final_url=source.url,
            retrieved_at=retrieved,
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
        ),
        signal=signal,
    )
    receipt = SourceReceiptInput(
        source_snapshot_id="snapshot_bhadrak",
        source_id=source.id,
        requested_url=source.url,
        final_url=source.url,
        retrieved_at="2026-07-21T14:00:00Z",
        status_code=200,
        content_type="text/html",
        byte_length=len(text.encode()),
        sha256=digest,
        access_path="live_origin",
    )
    rows = _signal_inputs(
        outcome, receipt, pipeline, source, registered_document=True
    )
    assert [row.district_id for row in rows] == ["OD-DIST-bhadrak"]
    assert [row.disease for row in rows] == ["malaria"]

    # A followed article gets no publisher allowance. The malaria sentence
    # still explicitly names Bhadrak, so sentence linkage keeps that real pair.
    desk = load_registry().get("sambad_district_bhadrak_or")
    assert desk.content_role == "index"
    desk_rows = _signal_inputs(
        outcome, receipt.model_copy(update={"source_id": desk.id}), pipeline, desk
    )
    assert len(desk_rows) == 1
    assert desk_rows[0].district_id == "OD-DIST-bhadrak"
    assert desk_rows[0].processing_state == "active_direct"

    no_place_text = "Malaria cases were reported by health officials."
    no_place_digest = hashlib.sha256(no_place_text.encode()).hexdigest()
    no_place_outcome = IngestionOutcome(
        receipt=outcome.receipt,
        signal=pipeline.process(
            Document(
                document_id=f"doc_{no_place_digest[:20]}",
                source_id=desk.id,
                canonical_url="https://sambad.in/an-article",
                retrieved_at=retrieved,
                content_type="text/html",
                text=no_place_text,
                sha256=no_place_digest,
            )
        ),
    )
    followed = _signal_inputs(
        no_place_outcome,
        receipt.model_copy(update={"source_id": desk.id}),
        pipeline,
        desk,
    )
    assert followed[0].district_id is None
    assert followed[0].processing_state == "ambiguous_entity_linkage"


def test_publisher_district_is_used_only_when_the_text_names_none() -> None:
    pipeline = IngestionPipeline.default()
    retrieved = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
    source = load_registry().get("district_khordha_health_or")
    assert source.district_id == "OD-DIST-khordha"

    def build(text: str) -> list:
        digest = hashlib.sha256(text.encode()).hexdigest()
        document = Document(
            document_id=f"doc_{digest[:20]}",
            source_id=source.id,
            canonical_url=source.url,
            retrieved_at=retrieved,
            content_type="text/html",
            text=text,
            sha256=digest,
        )
        outcome = IngestionOutcome(
            receipt=FetchReceipt(
                source_id=source.id,
                requested_url=source.url,
                final_url=source.url,
                retrieved_at=retrieved,
                status_code=200,
                content_type="text/html",
                byte_length=len(text.encode()),
                sha256=digest,
            ),
            signal=pipeline.process(document),
        )
        receipt = SourceReceiptInput(
            source_snapshot_id=f"snapshot_{digest[:20]}",
            source_id=source.id,
            requested_url=source.url,
            final_url=source.url,
            retrieved_at="2026-07-21T14:00:00Z",
            status_code=200,
            content_type="text/html",
            byte_length=len(text.encode()),
            sha256=digest,
            access_path="live_origin",
        )
        return _signal_inputs(
            outcome, receipt, pipeline, source, registered_document=True
        )

    # No district in the registered district health page: its publisher scope
    # supplies it. Followed newspaper links never receive this allowance.
    silent = build("ଡେଙ୍ଗୁ ରୋଗୀଙ୍କ ସଂଖ୍ୟା ବୃଦ୍ଧି ପାଇଛି ବୋଲି ସ୍ୱାସ୍ଥ୍ୟ ବିଭାଗ କହିଛି।")
    assert [row.district_id for row in silent] == ["OD-DIST-khordha"]

    # A district named in the text always wins over the publisher hint.
    explicit = build("ପୁରୀ ଜିଲ୍ଲାରେ ଡେଙ୍ଗୁ ମାମଲା ଚିହ୍ନଟ ହୋଇଛି।")
    assert [row.district_id for row in explicit] == ["OD-DIST-puri"]


def test_robots_disallow_blocks_the_job_before_any_content_request(
    tmp_path, monkeypatch
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'robots.sqlite3'}")
    source_id = "odisha_hfw_circulars_en"
    database.enqueue_job(
        source_id=source_id,
        kind="fetch",
        payload_ref="registered-link:https://health.odisha.gov.in/private/x",
        payload_hash="d" * 64,
        idempotency_key="robots-blocked",
    )
    runtime = CollectionRuntime(database)

    class _DenyAll:
        def evaluate(self, url: str) -> RobotsVerdict:  # noqa: ARG002
            return RobotsVerdict(allowed=False, state="disallowed")

    runtime.robots = _DenyAll()  # type: ignore[assignment]

    def explode(**kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("robots-disallowed URL must never be fetched")

    monkeypatch.setattr(
        "services.api.collection_runtime.ingest_registered_url", explode
    )
    job = database.claim_job(
        owner=runtime.owner,
        lease_seconds=300,
        kind="fetch",
        payload_prefix="registered-link:",
    )
    assert job is not None
    outcome = runtime._process(job)
    assert outcome["reason_code"] == "robots_disallowed"


def test_registry_covers_all_thirty_districts_in_three_languages() -> None:
    registry = load_registry()
    enabled = [source for source in registry.sources if source.enabled]
    assert len(enabled) > 7  # the pre-existing registry enabled seven routes
    languages = {language for source in enabled for language in source.languages}
    assert {"or", "hi", "en"} <= languages
    for language in ("or", "hi", "en"):
        assert sum(language in source.languages for source in enabled) >= 5
    districts = {source.district_id for source in enabled if source.district_id}
    assert len(districts) == 30
    hosts = {source.url.split("/")[2] for source in enabled}
    assert len(hosts) >= 40
    # Every enabled route must carry a real, dated verification receipt.
    for source in enabled:
        state = str(source.extra.get("availability_state", ""))
        assert state, f"{source.id} has no availability_state"
        if source.id != "idsp_weekly_outbreaks":
            assert "collector_http_200_" in state, f"{source.id}: {state}"
    for source in registry.sources:
        if not source.enabled:
            assert source.extra.get("disabled_reason"), source.id
