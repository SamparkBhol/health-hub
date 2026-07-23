"""Throughput, politeness and PDF-sandbox behaviour of the live collector."""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import UTC, datetime

from services.api.collection_runtime import CollectionRuntime
from services.api.database import Database
from workers.ingestion import robots as robots_module
from workers.ingestion.connectors import (
    AUTOMATIC_PDF_PARSE_BYTE_LIMIT,
    ingest_registered_url,
)
from workers.ingestion.models import FetchReceipt, FetchResult
from workers.ingestion.parse import ParsedText
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.registry import load_registry
from workers.ingestion.robots import HostRateLimiter, RobotsPolicy, RobotsVerdict


class _AllowAllRobots:
    def evaluate(self, url: str) -> RobotsVerdict:  # noqa: ARG002 - fixed verdict
        return RobotsVerdict(allowed=True, state="allowed", crawl_delay=0.0)


def _stub_robots_fetch(monkeypatch, bodies: dict[str, bytes]) -> None:
    def fake_fetch(url: str, **kwargs):  # noqa: ANN003, ANN202
        host = url.split("/")[2]
        if host not in bodies:
            raise robots_module.FetchError("http_404", "no robots.txt")
        body = bodies[host]
        return FetchResult(
            receipt=FetchReceipt(
                source_id="robots",
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
                status_code=200,
                content_type="text/plain",
                byte_length=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            ),
            body=body,
        )

    monkeypatch.setattr(robots_module, "fetch_url", fake_fetch)


def test_robots_policy_blocks_disallowed_paths_and_reads_crawl_delay(monkeypatch) -> None:
    _stub_robots_fetch(
        monkeypatch,
        {
            "strict.example.gov.in": (
                b"User-agent: *\nDisallow: /private/\nCrawl-delay: 4\n"
            ),
            # Several Indian government hosts answer /robots.txt with a themed
            # HTML error page and HTTP 200.  That is not a robots file.
            "themed.example.gov.in": b"<!DOCTYPE html><html><body>404</body></html>",
        },
    )
    policy = RobotsPolicy()
    allowed = policy.evaluate("https://strict.example.gov.in/en/health")
    assert allowed.allowed is True
    assert allowed.crawl_delay == 4.0  # noqa: PLR2004

    blocked = policy.evaluate("https://strict.example.gov.in/private/secret")
    assert blocked.allowed is False
    assert blocked.state == "disallowed"

    themed = policy.evaluate("https://themed.example.gov.in/en/health")
    assert themed.allowed is True
    assert themed.state == "robots_unavailable_html_error_page"

    missing = policy.evaluate("https://absent.example.gov.in/en/health")
    assert missing.allowed is True
    assert missing.state.startswith("robots_unavailable_http_404")


def test_host_rate_limiter_serialises_one_host_and_paces_it() -> None:
    limiter = HostRateLimiter()
    overlaps: list[int] = []
    active = 0
    guard = threading.Lock()
    starts: list[float] = []

    def worker() -> None:
        nonlocal active
        limiter.acquire("one.example.org", 0.25)
        with guard:
            active += 1
            overlaps.append(active)
            starts.append(time.monotonic())
        time.sleep(0.02)
        with guard:
            active -= 1
        limiter.release("one.example.org")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlaps) == 1, "requests to one origin must never overlap"
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    assert all(gap >= 0.2 for gap in gaps), gaps


def test_collector_fetches_distinct_hosts_in_parallel(tmp_path, monkeypatch) -> None:
    """Concurrency exists, and it is across hosts rather than within one."""

    database = Database(f"sqlite:///{tmp_path / 'parallel.sqlite3'}")
    runtime = CollectionRuntime(database)
    runtime.robots = _AllowAllRobots()  # type: ignore[assignment]
    sources = [
        source
        for source in load_registry().sources
        if source.enabled and source.url.startswith("https://sambad.in/district/")
    ][:6]
    assert len(sources) == 6  # noqa: PLR2004

    concurrent_now = 0
    peak = 0
    guard = threading.Lock()

    def fake_ingest(*, source_id: str, url: str, **kwargs):  # noqa: ANN003, ANN202
        nonlocal concurrent_now, peak
        with guard:
            concurrent_now += 1
            peak = max(peak, concurrent_now)
        time.sleep(0.05)
        with guard:
            concurrent_now -= 1
        raise robots_module.FetchError("http_503", "synthetic upstream")

    monkeypatch.setattr(
        "services.api.collection_runtime.ingest_registered_url", fake_ingest
    )

    jobs = []
    for index, source in enumerate(sources):
        job, _ = database.enqueue_job(
            source_id=source.id,
            kind="discover",
            payload_ref=f"registered-index:{source.id}",
            payload_hash=hashlib.sha256(source.id.encode()).hexdigest(),
            idempotency_key=f"parallel-{index}",
        )
        claimed = database.claim_job(
            owner=runtime.owner, lease_seconds=300, job_id=job["id"]
        )
        assert claimed is not None
        jobs.append(claimed)

    # All six share the host sambad.in, so they must run strictly one at a time.
    runtime._process_batch(jobs)
    assert peak == 1

    # Re-run the same six jobs keyed by their real per-source host: the district
    # collectorate portals are six different origins and do overlap.
    peak = 0
    portal_sources = [
        source
        for source in load_registry().sources
        if source.enabled and source.kind == "district_collectorate_health_page"
    ][:6]
    portal_jobs = []
    for index, source in enumerate(portal_sources):
        job, _ = database.enqueue_job(
            source_id=source.id,
            kind="discover",
            payload_ref=f"registered-index:{source.id}",
            payload_hash=hashlib.sha256(f"p{source.id}".encode()).hexdigest(),
            idempotency_key=f"parallel-portal-{index}",
        )
        claimed = database.claim_job(
            owner=runtime.owner, lease_seconds=300, job_id=job["id"]
        )
        assert claimed is not None
        portal_jobs.append(claimed)
    runtime._process_batch(portal_jobs)
    assert peak > 1


def test_ordinary_pdfs_are_parsed_without_a_pre_approved_digest(monkeypatch) -> None:
    """The digest allowlist no longer gates ordinary PDFs, only oversized ones."""

    pipeline = IngestionPipeline.default()
    body = b"%PDF-1.7 synthetic"
    digest = hashlib.sha256(body).hexdigest()

    def make_result(byte_length: int) -> FetchResult:
        return FetchResult(
            receipt=FetchReceipt(
                source_id="odisha_hfw_circulars_en",
                requested_url="https://health.odisha.gov.in/x.pdf",
                final_url="https://health.odisha.gov.in/x.pdf",
                retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
                status_code=200,
                content_type="application/pdf",
                byte_length=byte_length,
                sha256=digest,
            ),
            body=body,
        )

    monkeypatch.setattr(
        "workers.ingestion.connectors.parse_document",
        lambda *args, **kwargs: ParsedText(  # noqa: ARG005
            text="Dengue cases were reported in Khordha district.",
            parser="pdftotext_layout",
        ),
    )

    monkeypatch.setattr(
        "workers.ingestion.connectors.fetch_url",
        lambda *args, **kwargs: make_result(1024),  # noqa: ARG005
    )
    outcome = ingest_registered_url(
        registry=load_registry(),
        source_id="odisha_hfw_circulars_en",
        url="https://health.odisha.gov.in/x.pdf",
        pipeline=pipeline,
    )
    assert outcome.processing_state == "parsed_redacted_evidence"
    assert outcome.signal is not None
    assert outcome.signal.diseases == ("dengue",)

    # Above the automatic limit a person still decides, and the bytes are only
    # hashed until then.
    monkeypatch.setattr(
        "workers.ingestion.connectors.fetch_url",
        lambda *args, **kwargs: make_result(  # noqa: ARG005
            AUTOMATIC_PDF_PARSE_BYTE_LIMIT + 1
        ),
    )
    oversized = ingest_registered_url(
        registry=load_registry(),
        source_id="odisha_hfw_circulars_en",
        url="https://health.odisha.gov.in/x.pdf",
        pipeline=pipeline,
    )
    assert oversized.processing_state == "metadata_only_pdf_above_automatic_size_limit"
    assert oversized.signal is None

    approved = ingest_registered_url(
        registry=load_registry(),
        source_id="odisha_hfw_circulars_en",
        url="https://health.odisha.gov.in/x.pdf",
        pipeline=pipeline,
        approved_pdf_sha256s=frozenset({digest}),
    )
    assert approved.processing_state == "parsed_redacted_evidence"


def test_idsp_pdf_with_zero_odisha_rows_never_falls_through_to_news_extraction(
    monkeypatch,
) -> None:
    body = b"%PDF-1.7 synthetic national report"
    url = "https://idsp.mohfw.gov.in/WriteReadData/l892s/week13.pdf"
    result = FetchResult(
        receipt=FetchReceipt(
            source_id="idsp_weekly_outbreaks",
            requested_url=url,
            final_url=url,
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            status_code=200,
            content_type="application/pdf",
            byte_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
        ),
        body=body,
    )
    monkeypatch.setattr(
        "workers.ingestion.connectors.IdspConnector.fetch_report",
        lambda self, requested_url: result,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "workers.ingestion.connectors.parse_document",
        lambda *args, **kwargs: ParsedText(  # noqa: ARG005
            text=(
                "Dengue and malaria were discussed nationally. "
                "Bhadrak district submitted an unrelated status note."
            ),
            parser="pdftotext_layout",
        ),
    )
    outcome = ingest_registered_url(
        registry=load_registry(),
        source_id="idsp_weekly_outbreaks",
        url=url,
        pipeline=IngestionPipeline.default(),
    )
    assert outcome.processing_state == "positive_only_official_catalogue"
    assert outcome.catalogue_rows == ()
    assert outcome.signal is None


def test_cleartext_links_are_dropped_at_discovery(tmp_path) -> None:
    """An http:// row can never be fetched, so it must never become a job.

    Queuing one burned all three of its retries and then dead-lettered it, which
    is exactly what happened on the district collectorate portals.
    """

    from workers.ingestion.connectors import discover_registered_links

    body = b"""<html><body><ul>
      <li><a href="http://puri.odisha.gov.in/en/dengue-advisory">Dengue advisory</a></li>
      <li><a href="https://puri.odisha.gov.in/en/malaria-advisory">Malaria advisory</a></li>
    </ul></body></html>"""
    source = load_registry().get("district_puri_health_en")
    links = discover_registered_links(
        body,
        index_url="https://puri.odisha.gov.in/en/departments/health",
        source=source,
    )
    assert [link.url for link in links] == [
        "https://puri.odisha.gov.in/en/malaria-advisory"
    ]
