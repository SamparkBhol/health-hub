"""Bounded in-process collector backed by durable PostgreSQL/SQLite jobs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, cast

from apscheduler.schedulers.background import BackgroundScheduler

from packages.contracts.api import (
    LIVE_EVIDENCE_PLACEHOLDER,
    LIVE_EVIDENCE_REDACTION_STATE,
    RedactedSignalInput,
    SourceReceiptInput,
)
from workers.ingestion.assertions import classify_assertion
from workers.ingestion.connectors import (
    IngestionOutcome,
    ingest_registered_url,
    is_navigation_url,
)
from workers.ingestion.parse import ParseError, tesseract_pdf_ocr
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.registry import SourceSpec, load_registry
from workers.ingestion.robots import HostRateLimiter, RobotsPolicy, host_of
from workers.ingestion.safe_fetch import FetchError, crawler_contact

from .database import Database, RepositoryConflict, RepositoryNotFound

COLLECTION_WINDOW_SECONDS = 6 * 60 * 60
# A two-link cap could never reach 30 districts: an index page yields dozens of
# notices and the queue drained two of them per source, for ever.
MAX_DISCOVERED_LINKS = 24
# Upper bound on how much of one index page is remembered per discovery pass.
MAX_REGISTERED_LINKS_PER_DISCOVERY = 60
JOB_LEASE_SECONDS = 300
JOB_WORK_BUDGET_SECONDS = 270
# Throughput knobs.  Concurrency is across *distinct origin hosts* only; the
# per-host limiter below still serialises and paces every individual site.
DEFAULT_JOBS_PER_TICK = 40
MAXIMUM_JOBS_PER_TICK = 200
DEFAULT_FETCH_WORKERS = 8
MAXIMUM_FETCH_WORKERS = 16
# Per-host serialisation means a bucket's wall-clock cost is the sum of its
# jobs.  Keeping each bucket well inside one lease stops a prolific origin from
# holding leases it cannot finish, which would waste the fetches already made.
MAX_JOBS_PER_HOST_PER_TICK = 10
SCHEDULER_INTERVAL_SECONDS = 120
TICK_WALL_CLOCK_BUDGET_SECONDS = 900.0
HEALTH_LABEL_MARKERS = (
    "health",
    "outbreak",
    "disease",
    "hospital",
    "public health",
    "स्वास्थ्य",
    "अस्पताल",
    "ସ୍ୱାସ୍ଥ୍ୟ",
    "ଡାକ୍ତରଖାନା",
)

AssertionValue = Literal["affirmed", "not_affirmed", "speculative", "non_current"]

# A `dead` job is not automatically a broken source.  Global URL-ownership
# dedup retires the losing copy of a link a different route already owns, and
# canonicalisation retires a job that still names a pre-canonical URL.  Both are
# housekeeping and must not be read as crawl failures; everything else is one.
DEDUPLICATION_RETIREMENT_CODES: frozenset[str] = frozenset(
    {"DUPLICATE_URL_OWNER", "URL_CANONICALIZED"}
)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _receipt_input(outcome: IngestionOutcome) -> SourceReceiptInput:
    receipt = outcome.receipt
    seed = (
        f"{receipt.source_id}\x1f{receipt.retrieved_at.isoformat()}\x1f{receipt.sha256}"
    ).encode()
    snapshot_id = f"snapshot_{hashlib.sha256(seed).hexdigest()[:32]}"
    return SourceReceiptInput(
        source_snapshot_id=snapshot_id,
        source_id=receipt.source_id,
        requested_url=receipt.requested_url,
        final_url=receipt.final_url,
        retrieved_at=_utc_text(receipt.retrieved_at),
        status_code=receipt.status_code,
        content_type=receipt.content_type,
        byte_length=receipt.byte_length,
        sha256=receipt.sha256,
        access_path=receipt.access_path,
        archive_timestamp=receipt.archive_timestamp,
        archive_digest=receipt.archive_digest,
        fallback_reason=receipt.fallback_reason,
        is_fixture=False,
    )


def _signal_inputs(
    outcome: IngestionOutcome,
    receipt: SourceReceiptInput,
    pipeline: IngestionPipeline,
    source: SourceSpec | None = None,
    *,
    registered_document: bool = False,
) -> list[RedactedSignalInput]:
    """Turn one document into conservatively linked disease/place signals.

    A disease and district must co-occur in one sentence-like block.  This
    prevents a national bulletin or multi-item PDF from assigning every disease
    in the document to the one district mentioned elsewhere.  Publisher scope
    may fill a missing place only for the registered document itself; followed
    links never inherit the district of the section that discovered them.
    """

    signal = outcome.signal
    if signal is None:
        return []
    if not signal.diseases:
        return []
    publisher_district = (
        source.district_id
        if registered_document and source is not None and source.content_role == "document"
        else None
    )
    linked: list[tuple[str, str, AssertionValue]] = []
    unresolved: set[str] = set()
    # Full stops, Indic danda and newlines are deliberately conservative
    # boundaries.  A block with multiple diseases and multiple districts stays
    # unresolved rather than guessing argument roles.
    blocks = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?।॥\r\n]+(?:[.!?।॥]+|$)", signal.redacted_evidence)
        if match.group(0).strip()
    ]
    for block in blocks:
        diseases = pipeline.diseases.find(block)
        if not diseases:
            continue
        districts = {match.district_id for match in pipeline.gazetteer.resolve(block)}
        assertion = cast(
            AssertionValue,
            classify_assertion(block, as_of=signal.retrieved_at.date()).value,
        )
        district_id: str | None = None
        if len(districts) == 1:
            district_id = next(iter(districts))
        elif publisher_district and (
            not districts or publisher_district in districts
        ):
            district_id = publisher_district
        if district_id is None:
            unresolved.update(diseases)
            continue
        linked.extend((disease, district_id, assertion) for disease in diseases)

    linked = list(dict.fromkeys(linked))
    linked_diseases = {disease for disease, _, _ in linked}
    unresolved.difference_update(linked_diseases)
    targets: list[tuple[str | None, str | None, AssertionValue, bool]] = [
        (disease, district_id, assertion, False)
        for disease, district_id, assertion in linked
    ]
    if unresolved or not targets:
        targets.append(
            (
                next(iter(unresolved)) if len(unresolved) == 1 else None,
                None,
                cast(AssertionValue, signal.assertion.value),
                True,
            )
        )

    base_processing_state: Literal[
        "active_direct",
        "language_review_required",
        "privacy_review_required",
        "ambiguous_entity_linkage",
    ]
    if signal.coverage_state.value == "active_direct":
        base_processing_state = "active_direct"
    elif signal.coverage_state.value == "language_review_required":
        base_processing_state = "language_review_required"
    else:
        base_processing_state = "privacy_review_required"
    redaction_state: Literal[
        "heuristic_unvalidated",
        "content_not_retained_unvalidated_pii",
    ]
    retain_synthetic_fixture = receipt.is_fixture and signal.is_synthetic_fixture
    if retain_synthetic_fixture:
        evidence_start, evidence_text = _bounded_evidence_span(signal, pipeline)
        redaction_state = "heuristic_unvalidated"
    else:
        # Live source text is used only transiently by extraction.  Heuristic
        # multilingual PERSON detection is not accurate enough to make any
        # source-language span safe to retain in the competition profile.
        evidence_start = 0
        evidence_text = LIVE_EVIDENCE_PLACEHOLDER
        redaction_state = LIVE_EVIDENCE_REDACTION_STATE
    inputs: list[RedactedSignalInput] = []
    for disease, district_id, assertion, ambiguous in targets:
        processing_state = (
            "ambiguous_entity_linkage" if ambiguous else base_processing_state
        )
        eligible = bool(
            not ambiguous
            and disease
            and district_id
            and assertion == "affirmed"
            and processing_state == "active_direct"
        )
        if retain_synthetic_fixture:
            suffix = hashlib.sha256(
                f"{disease or 'ambiguous'}\x1f{district_id or ''}\x1f{assertion}".encode()
            ).hexdigest()[:8]
            signal_id = f"{signal.signal_id}_{suffix}"
        else:
            identity_material = "\x1f".join(
                (
                    receipt.source_id,
                    receipt.source_snapshot_id,
                    disease or "ambiguous",
                    district_id or "",
                    assertion,
                    processing_state,
                    signal.language.value,
                )
            )
            signal_id = f"sig_{hashlib.sha256(identity_material.encode()).hexdigest()[:32]}"
        inputs.append(
            RedactedSignalInput(
                signal_id=signal_id,
                source_id=receipt.source_id,
                source_snapshot_id=receipt.source_snapshot_id,
                district_id=district_id,
                disease=disease,
                assertion=assertion,
                evidence_text=evidence_text,
                evidence_start=evidence_start,
                evidence_end=evidence_start + len(evidence_text),
                content_sha256=hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
                retrieved_at=receipt.retrieved_at,
                event_review_eligible=eligible,
                processing_state=processing_state,
                redaction_state=redaction_state,
                language=signal.language.value,
                extractor_version="rules-sentence-link-v2",
            )
        )
    return inputs


def _bounded_evidence_span(
    signal: Any,
    pipeline: IngestionPipeline,
    *,
    maximum_characters: int = 2400,
) -> tuple[int, str]:
    """Select a deterministic disease/place-centred synthetic-fixture excerpt.

    Offsets are in the canonical fixture text. Live collection never calls
    this retention helper; it persists the fixed non-retention placeholder.
    """

    text = signal.redacted_evidence
    if len(text) <= maximum_characters:
        return 0, text
    folded = text.casefold()
    anchors: list[int] = []
    for match in signal.districts:
        position = folded.find(match.matched_alias.casefold())
        if position >= 0:
            anchors.append(position)
    for disease in signal.diseases:
        for alias in pipeline.diseases.terms.get(disease, ()):
            position = folded.find(alias.casefold())
            if position >= 0:
                anchors.append(position)
    anchor = min(anchors) if anchors else 0
    start = max(0, anchor - maximum_characters // 3)
    end = min(len(text), start + maximum_characters)
    start = max(0, end - maximum_characters)
    # Prefer human-readable boundaries without exceeding the hard contract.
    if start:
        boundary = max(
            text.rfind("\n", start, min(start + 200, end)),
            text.rfind(". ", start, min(start + 200, end)),
        )
        if boundary >= start:
            start = boundary + 1
    if end < len(text):
        candidates = [
            value
            for value in (text.rfind("\n", start, end), text.rfind(". ", start, end))
            if value > start
        ]
        if candidates:
            end = max(candidates) + 1
    excerpt = text[start:end].strip()
    leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
    return start + leading_trim, excerpt


class CollectionRuntime:
    """Schedule and execute a small number of allowlisted collection jobs."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.registry = load_registry()
        self.pipeline = IngestionPipeline.default()
        self.owner = f"collector-{uuid.uuid4().hex[:12]}"
        self.scheduler: BackgroundScheduler | None = None
        self.robots = RobotsPolicy()
        self.host_limiter = HostRateLimiter()
        self._rotation = 0

    @property
    def enabled(self) -> bool:
        """Whether the in-process scheduler runs.

        Live collection is the product, so it is on unless an operator opts out.
        """

        return os.getenv("ENABLE_IN_PROCESS_SCHEDULER", "true").casefold() != "false"

    @property
    def live_collection_enabled(self) -> bool:
        return os.getenv("LIVE_COLLECTION_ENABLED", "true").casefold() != "false"

    @property
    def crawler_contact(self) -> str:
        return crawler_contact()

    @property
    def contact_configured(self) -> bool:
        """Always true: a project-identifying contact is used when none is set.

        The former behaviour -- withhold every request until an operator export
        exists -- meant the shipped product crawled nothing and showed fixtures.
        Identifying the crawler is the actual courtesy owed to an origin, and
        `workers.ingestion.safe_fetch.crawler_contact` guarantees one.
        """

        return True

    @property
    def fetch_workers(self) -> int:
        try:
            value = int(os.getenv("COLLECTOR_FETCH_WORKERS", str(DEFAULT_FETCH_WORKERS)))
        except ValueError:
            value = DEFAULT_FETCH_WORKERS
        return max(1, min(value, MAXIMUM_FETCH_WORKERS))

    @property
    def jobs_per_tick(self) -> int:
        try:
            value = int(os.getenv("COLLECTOR_JOBS_PER_TICK", str(DEFAULT_JOBS_PER_TICK)))
        except ValueError:
            value = DEFAULT_JOBS_PER_TICK
        return max(1, min(value, MAXIMUM_JOBS_PER_TICK))

    @property
    def approved_pdf_sha256s(self) -> frozenset[str]:
        values = {
            value.strip().lower()
            for value in os.getenv("APPROVED_PDF_SHA256S", "").split(",")
            if value.strip()
        }
        return frozenset(value for value in values if len(value) == 64)

    def start(self) -> None:
        if not self.enabled or not self.live_collection_enabled:
            return
        scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
        scheduler.add_job(
            self.tick,
            "interval",
            seconds=SCHEDULER_INTERVAL_SECONDS,
            id="registered-source-collector",
            coalesce=True,
            max_instances=1,
            replace_existing=True,
        )
        scheduler.start()
        self.scheduler = scheduler

    def stop(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None

    def dead_job_reason_counts(self) -> dict[str, int]:
        """Histogram of `last_error_code` across dead jobs, largest bucket first.

        Without this the operations view showed one `dead` number in which a
        genuine `http_404` or `robots_disallowed` was indistinguishable from a
        deduplication retirement, so a totally broken origin looked like tidy
        housekeeping.
        """

        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT last_error_code AS reason_code, COUNT(*) AS total
                FROM job
                WHERE state='dead'
                GROUP BY last_error_code
                """
            ).fetchall()
        counts = {
            str(row["reason_code"] or "unrecorded"): int(row["total"]) for row in rows
        }
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def status(self) -> dict[str, Any]:
        enabled_sources = [source for source in self.registry.sources if source.enabled]
        hosts = {host for source in enabled_sources for host in source.allowed_hosts}
        languages: set[str] = set()
        districts: set[str] = set()
        for source in enabled_sources:
            languages.update(source.languages)
            if source.district_id:
                districts.add(source.district_id)
        dead_reasons = self.dead_job_reason_counts()
        deduplicated = sum(
            total
            for code, total in dead_reasons.items()
            if code in DEDUPLICATION_RETIREMENT_CODES
        )
        queue = dict(self.database.job_backlog_counts())
        queue["dead_deduplication_retired"] = deduplicated
        queue["dead_failed"] = max(0, int(queue.get("dead", 0)) - deduplicated)
        return {
            "configured": self.enabled,
            "contact_configured": self.contact_configured,
            "live_collection_enabled": self.live_collection_enabled,
            "crawler_contact": self.crawler_contact,
            "running": bool(self.scheduler and self.scheduler.running),
            "authoritative_scheduler": "in_process_plus_github_actions",
            "runtime_role": "bounded_in_process_worker_while_awake",
            "queue": queue,
            "dead_job_reasons": dead_reasons,
            "collection_failure_visible": bool(queue["dead_failed"]),
            "api_jobs_per_tick": self.jobs_per_tick,
            "fetch_workers": self.fetch_workers,
            "scheduler_interval_seconds": SCHEDULER_INTERVAL_SECONDS,
            "scheduled_trigger_round_limit": 10,
            "enabled_routes": len(enabled_sources),
            "registered_routes": len(self.registry.sources),
            "distinct_hosts": len(hosts),
            "route_languages": sorted(languages),
            "district_scoped_routes": len(districts),
        }

    def tick(self, *, maximum_jobs: int | None = None) -> dict[str, Any]:
        if not self.live_collection_enabled:
            return {
                "state": "withheld",
                "reason_code": "LIVE_COLLECTION_DISABLED",
                "enqueued": 0,
                "processed": [],
            }
        budget = self.jobs_per_tick if maximum_jobs is None else maximum_jobs
        budget = max(1, min(budget, MAXIMUM_JOBS_PER_TICK))
        enqueued = self._enqueue_index_jobs()
        claimed = self._claim_batch(budget)
        processed = self._process_batch(claimed)
        return {
            "state": "completed",
            "enqueued": enqueued,
            "processed": processed,
        }

    def _claimable_link_hosts(self) -> tuple[str, ...]:
        """Every origin a discovered-link job can legitimately name.

        A link job's `payload_ref` is the absolute URL that discovery accepted,
        and discovery accepts any host on the source's allow-list -- `sambad.in`
        publishes articles under `www.sambad.in`, `osdma.org` under
        `www.osdma.org`.  Building the claim prefixes from the registered URL's
        host alone left those jobs with no prefix that ever matched, so they sat
        queued for ever.  Hosts are ordered so the registered origins come
        first and the extra allow-list aliases follow, keeping the existing
        breadth-first rotation stable.
        """

        registered: list[str] = []
        aliases: list[str] = []
        for source in self.registry.sources:
            if not source.enabled:
                continue
            primary = host_of(source.url)
            if primary and primary not in registered:
                registered.append(primary)
            for host in source.allowed_hosts:
                normalised = host.strip().casefold()
                if normalised and normalised not in aliases:
                    aliases.append(normalised)
        return tuple(registered) + tuple(
            host for host in aliases if host not in registered
        )

    def _claim_batch(self, budget: int) -> list[dict[str, Any]]:
        """Lease up to `budget` jobs, discovery surfaces first, breadth-first by host.

        Jobs are created in per-source bursts, so a naive "claim the oldest N"
        leased two hundred jobs belonging to three origins.  Because a host is
        crawled serially, most of that lease could not be used inside one tick
        and simply expired.  Claiming is therefore rotated over origins: every
        host contributes at most `MAX_JOBS_PER_HOST_PER_TICK`, which is exactly
        what one tick can actually execute, and 30 districts advance together
        instead of one newspaper monopolising the batch.
        """

        claimed: list[dict[str, Any]] = []
        by_host: dict[str, list[SourceSpec]] = {}
        for source in self.registry.sources:
            if source.enabled:
                by_host.setdefault(host_of(source.url), []).append(source)
        link_hosts = self._claimable_link_hosts()

        # Discovery surfaces: one registered index per source.  The starting
        # offset rotates per tick so that a host with more routes than the
        # per-tick cap (a newspaper with 30 district desks) does not always
        # present the same first ten.
        self._rotation += 1
        widest = max((len(items) for items in by_host.values()), default=0)
        for depth in range(min(MAX_JOBS_PER_HOST_PER_TICK, widest)):
            for sources in by_host.values():
                if len(claimed) >= budget:
                    return claimed
                if depth >= len(sources):
                    continue
                source = sources[(depth + self._rotation) % len(sources)]
                job = self.database.claim_job(
                    owner=self.owner,
                    lease_seconds=JOB_LEASE_SECONDS,
                    kind="discover",
                    payload_prefix=f"registered-index:{source.id}",
                )
                if job is not None:
                    claimed.append(job)

        # Detail links: a link job's payload_ref carries its absolute URL, so
        # the origin can be selected directly.
        counts = dict.fromkeys(link_hosts, 0)
        for _ in range(MAX_JOBS_PER_HOST_PER_TICK):
            progressed = False
            for host in link_hosts:
                if len(claimed) >= budget:
                    return claimed
                if counts[host] >= MAX_JOBS_PER_HOST_PER_TICK:
                    continue
                job = self.database.claim_job(
                    owner=self.owner,
                    lease_seconds=JOB_LEASE_SECONDS,
                    kind="fetch",
                    payload_prefix=f"registered-link:https://{host}/",
                )
                if job is not None:
                    claimed.append(job)
                    counts[host] += 1
                    progressed = True
            if not progressed:
                break
        return claimed

    def _process_batch(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run leased jobs, in parallel across distinct hosts only.

        Jobs are bucketed by origin host and each bucket is handled by one
        worker in order, so a site sees strictly sequential, paced requests
        while unrelated sites proceed at the same time.
        """

        if not jobs:
            return []
        buckets: dict[str, list[dict[str, Any]]] = {}
        for job in jobs:
            buckets.setdefault(self._job_host(job), []).append(job)
        started_at = time.monotonic()
        tick_deadline = started_at + TICK_WALL_CLOCK_BUDGET_SECONDS
        # A bucket is serial, so its cost is the sum of its jobs.  Stop well
        # inside the lease: work started after that would be completed against a
        # lease this worker no longer holds, throwing the fetch away.
        lease_deadline = started_at + JOB_LEASE_SECONDS * 0.8
        deadline = min(tick_deadline, lease_deadline)

        def run_bucket(bucket: list[dict[str, Any]]) -> list[dict[str, Any]]:
            outcomes: list[dict[str, Any]] = []
            for index, job in enumerate(bucket):
                if index >= MAX_JOBS_PER_HOST_PER_TICK or time.monotonic() > deadline:
                    # Leave the remaining leases to expire rather than failing
                    # them: an abandoned lease is re-claimable, while a recorded
                    # failure consumes one of the job's three retries.
                    outcomes.append(
                        {
                            "job_id": job["id"],
                            "source_id": job["source_id"],
                            "state": "deferred_to_next_tick",
                        }
                    )
                    continue
                outcomes.append(self._process(job))
            return outcomes

        if len(buckets) == 1:
            return run_bucket(next(iter(buckets.values())))
        results: list[dict[str, Any]] = []
        workers = min(self.fetch_workers, len(buckets))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for outcomes in pool.map(run_bucket, list(buckets.values())):
                results.extend(outcomes)
        return results

    def _job_host(self, job: dict[str, Any]) -> str:
        payload_ref = str(job["payload_ref"] or "")
        if payload_ref.startswith("registered-link:"):
            return host_of(payload_ref.removeprefix("registered-link:"))
        try:
            source = self.registry.get(str(job["source_id"]))
        except Exception:  # noqa: BLE001 - a disabled/unknown source is handled in _process
            return str(job["source_id"])
        return host_of(source.url)


    def _enqueue_index_jobs(self) -> int:
        bucket = int(time.time()) // COLLECTION_WINDOW_SECONDS
        created = 0
        for source in self.registry.sources:
            if not source.enabled:
                continue
            created += self._enqueue_pending_links(source)
            key = f"scheduled-index:{source.id}:{bucket}"
            digest = hashlib.sha256(key.encode()).hexdigest()
            try:
                _, replayed = self.database.enqueue_job(
                    source_id=source.id,
                    kind="discover",
                    payload_ref=f"registered-index:{source.id}",
                    payload_hash=digest,
                    idempotency_key=key,
                )
            except (RepositoryConflict, RepositoryNotFound):
                continue
            created += int(not replayed)
        return created

    def _process(self, job: dict[str, Any]) -> dict[str, Any]:
        source = self.registry.get(job["source_id"])
        payload_ref = str(job["payload_ref"] or "")
        is_registered_index = payload_ref == f"registered-index:{source.id}"
        if is_registered_index:
            url = source.url
            # A registered `document` route is a published page in its own
            # right, so its own text is evidence *and* its links are followed.
            is_index = source.is_index_only
        elif payload_ref.startswith("registered-link:"):
            url = payload_ref.removeprefix("registered-link:")
            is_index = False
        else:
            return self._fail(job, source, "UNSUPPORTED_PAYLOAD_REF", retryable=False)
        verdict = self.robots.evaluate(url)
        if not verdict.allowed:
            return self._fail(job, source, f"robots_{verdict.state}", retryable=False)
        work_deadline = time.monotonic() + JOB_WORK_BUDGET_SECONDS

        def lease_bounded_ocr(body: bytes, language_hint: str | None = None):  # noqa: ANN202
            remaining = int(work_deadline - time.monotonic())
            if remaining < 1:
                raise ParseError("collection job OCR budget exhausted")
            return tesseract_pdf_ocr(
                body,
                language_hint,
                timeout_seconds=remaining,
            )

        host = host_of(url)
        interval = float(
            verdict.crawl_delay
            if verdict.crawl_delay is not None
            else source.minimum_interval_seconds
        )
        try:
            self.host_limiter.acquire(host, interval)
            try:
                outcome = ingest_registered_url(
                    registry=self.registry,
                    source_id=source.id,
                    url=url,
                    pipeline=self.pipeline,
                    ocr_hook=lease_bounded_ocr,
                    approved_pdf_sha256s=self.approved_pdf_sha256s,
                )
            finally:
                self.host_limiter.release(host)
            receipt = _receipt_input(outcome)
            # Index/listing pages are discovery surfaces, not evidence items.
            signals = (
                []
                if is_index
                else _signal_inputs(
                    outcome,
                    receipt,
                    self.pipeline,
                    source,
                    registered_document=(
                        is_registered_index and source.content_role == "document"
                    ),
                )
            )
            self.database.complete_job(
                job_id=job["id"],
                owner=self.owner,
                fencing_token=job["fencing_token"],
                idempotency_key=f"collection-complete:{job['id']}:{receipt.sha256}",
                receipt=receipt,
                signals=signals,
                link_disposition=(
                    "pending_approval"
                    if outcome.processing_state.startswith("metadata_only_")
                    else "fetched"
                ),
            )
            catalogue_events: list[dict[str, Any]] = []
            for row in outcome.catalogue_rows:
                districts = self.pipeline.gazetteer.resolve(row.source_text)
                diseases = self.pipeline.diseases.find(row.source_text)
                catalogue_events.append(
                    {
                        "outbreak_id": row.outbreak_id,
                        "year": row.year,
                        "week": row.week,
                        "district_code": row.district_code,
                        "district_id": (
                            districts[0].district_id if len(districts) == 1 else None
                        ),
                        "disease": diseases[0] if len(diseases) == 1 else None,
                        "authority_status": row.authority_status,
                        "positive_only_catalogue": True,
                        "missing_weeks_are_not_zero": True,
                    }
                )
            catalogue_count = self.database.upsert_catalogue_events(
                source_id=source.id,
                source_snapshot_id=receipt.source_snapshot_id,
                events=catalogue_events,
            )
            self.database.mark_source_collection(source.id, succeeded=True)
            discovered = (
                self._enqueue_discovered(source, outcome) if is_registered_index else 0
            )
            return {
                "job_id": job["id"],
                "source_id": source.id,
                "state": "completed",
                "signal_count": len(signals),
                "catalogue_event_count": catalogue_count,
                "discovered_jobs": discovered,
                "snapshot_id": receipt.source_snapshot_id,
                "content_sha256": receipt.sha256,
                "processing_state": (
                    "index_discovery_only" if is_index else outcome.processing_state
                ),
            }
        except FetchError as exc:
            retryable = exc.code in {"network_error", "dns_failed"} or exc.code.startswith(
                "http_5"
            )
            return self._fail(job, source, exc.code, retryable=retryable)
        except ParseError as exc:
            return self._fail(job, source, type(exc).__name__, retryable=False)
        except Exception as exc:  # noqa: BLE001 - typed failure is persisted; no source text logged
            return self._fail(job, source, type(exc).__name__, retryable=False)

    def _enqueue_discovered(self, source: SourceSpec, outcome: IngestionOutcome) -> int:
        ranked = sorted(
            outcome.discovered_links,
            key=lambda link: (
                not bool(self.pipeline.diseases.find(link.label)),
                not any(marker in link.label.casefold() for marker in HEALTH_LABEL_MARKERS),
                link.content_hint != "application/pdf",
                -link.score,
                link.url,
            ),
        )
        # A section route may admit a positively scored article even when the
        # headline omits a canonical disease term. Zero-score links are site
        # furniture in practice (category, crime, lifestyle, author pages) and
        # previously contaminated every district route with identical content.
        follow_section = str(source.extra.get("link_policy", "health_only")) == "section"
        eligible = [
            link
            for link in ranked
            if not is_navigation_url(link.url)
            and (
                self.pipeline.diseases.find(link.label)
                or any(marker in link.label.casefold() for marker in HEALTH_LABEL_MARKERS)
                or (
                    link.content_hint == "application/pdf"
                    and "pdf" in source.kind.casefold()
                )
                or (follow_section and link.score > 0.0 and link.label.strip())
            )
        ][:MAX_REGISTERED_LINKS_PER_DISCOVERY]
        self.database.register_discovered_links(
            source_id=source.id,
            links=[
                {
                    "url": link.url,
                    "label": link.label,
                    "content_hint": link.content_hint,
                }
                for link in eligible
            ],
        )
        return self._enqueue_pending_links(source)

    def _enqueue_pending_links(self, source: SourceSpec) -> int:
        active = sum(
            item["state"] == "queued"
            for item in self.database.list_discovered_links(source.id)
        )
        available_slots = max(0, MAX_DISCOVERED_LINKS - active)
        selected = self.database.reserve_discovered_links(
            source_id=source.id,
            limit=available_slots,
            approved_content_sha256s=self.approved_pdf_sha256s,
        )
        created = 0
        for link in selected:
            try:
                _, replayed = self.database.enqueue_reserved_discovered_link(
                    source_id=source.id,
                    url=link["url"],
                )
            except (RepositoryConflict, RepositoryNotFound):
                self.database.release_discovered_link(
                    source_id=source.id,
                    url=link["url"],
                )
                continue
            created += int(not replayed)
        return created

    def _fail(
        self,
        job: dict[str, Any],
        source: SourceSpec,
        reason_code: str,
        *,
        retryable: bool,
    ) -> dict[str, Any]:
        try:
            failed = self.database.fail_job(
                job_id=job["id"],
                owner=self.owner,
                fencing_token=job["fencing_token"],
                reason_code=reason_code[:100],
                retryable=retryable,
            )
            state = failed["state"]
        except RepositoryConflict:
            state = "lease_lost"
        self.database.mark_source_collection(
            source.id,
            succeeded=False,
            error_code=reason_code[:100],
        )
        return {
            "job_id": job["id"],
            "source_id": source.id,
            "state": state,
            "reason_code": reason_code[:100],
        }
