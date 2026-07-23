"""Regressions for the verified acquisition defects.

Each test pins one behaviour that was measured as broken against the live
corpus: an https->http redirect that dropped 23 district pages, a masthead that
suppressed most district-linked signals, an OCR language taken from the route
config, a claim prefix built from one host per source, a dead-job counter that
mixed failures with dedup retirements, and a Hindi tier with no district scope.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from services.api.collection_runtime import CollectionRuntime
from services.api.database import Database
from workers.ingestion.models import Document
from workers.ingestion.parse import (
    OCR_LANGUAGES,
    parse_html,
    tesseract_languages,
)
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.redact import redact_pii
from workers.ingestion.registry import load_registry
from workers.ingestion.safe_fetch import FetchError, fetch_url

# The Odisha district portals wrap the whole page in an unclosed
# `div.megamenu-nav`, and every one of them carries the Chief Minister in a
# `ul.header-extra-info` masthead.  This is that shape, minimised.
_PORTAL_HTML = b"""
<html><head><title>Health | Khordha</title></head><body>
<div class="megamenu-nav">
  <ul class="header-extra-info">
    <li><div class="chief-minister"><div class="cm-name">Shri Mohan Charan Majhi</div>
    <div class="cm-desig">Chief Minister, Odisha</div></div></li>
  </ul>
  <nav class="main-navigation"><ul role="menu"><li>Home</li><li>Departments</li></ul></nav>
  <div class="content">
    <p>The Khordha district health office reported nine dengue cases this week.</p>
    <p>Vector control teams surveyed the affected wards and issued an advisory
    to residents about stagnant water around the municipal ward offices.</p>
  </div>
</body></html>
"""


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status
        self.headers = headers
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_bytes(self):  # noqa: ANN202 - httpx streaming shape
        yield self.body


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[int, dict[str, str], bytes]],
    requests: list[str],
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            return None

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stream(self, method, url, *, headers, extensions):  # noqa: ANN001, ANN202
            assert method == "GET"
            requests.append(f"{headers['Host']}{url.split(':443', 1)[1]}")
            return _Response(*responses.pop(0))

    monkeypatch.setattr("workers.ingestion.safe_fetch.httpx.Client", Client)


def _resolver(host: str, port: int, **kwargs: object):  # noqa: ANN202, ARG001
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def test_origin_scheme_downgrade_is_followed_over_tls_and_recorded() -> None:
    """A 301 to the same host over http reaches the page and stays auditable."""

    requests: list[str] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_transport(
            monkeypatch,
            [
                (
                    301,
                    {"location": "http://balangir.odisha.gov.in/or/notices/announcement"},
                    b"",
                ),
                (200, {"content-type": "text/html"}, b"<html>ok</html>"),
            ],
            requests,
        )
        result = fetch_url(
            "https://balangir.odisha.gov.in/notices/announcements",
            source_id="downgrade-test",
            allowed_hosts=("balangir.odisha.gov.in",),
            resolver=_resolver,
        )
    finally:
        monkeypatch.undo()

    assert result.body == b"<html>ok</html>"
    # The second hop is fetched over TLS, never in the clear.
    assert requests == [
        "balangir.odisha.gov.in/notices/announcements",
        "balangir.odisha.gov.in/or/notices/announcement",
    ]
    receipt = result.receipt
    assert receipt.final_url == "https://balangir.odisha.gov.in/or/notices/announcement"
    assert receipt.scheme_downgraded is True
    assert receipt.scheme_downgrades == (
        "http://balangir.odisha.gov.in/or/notices/announcement",
    )
    assert receipt.access_path == (
        "live_origin+origin_scheme_downgrade_refetched_over_https"
    )


def test_scheme_downgrade_to_the_same_url_is_a_typed_failure_not_a_loop() -> None:
    requests: list[str] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_transport(
            monkeypatch,
            [(301, {"location": "http://balangir.odisha.gov.in/health"}, b"")],
            requests,
        )
        with pytest.raises(FetchError) as error:
            fetch_url(
                "https://balangir.odisha.gov.in/health",
                source_id="downgrade-loop",
                allowed_hosts=("balangir.odisha.gov.in",),
                resolver=_resolver,
            )
    finally:
        monkeypatch.undo()
    assert error.value.code == "scheme_downgrade_loop"


def test_cleartext_redirect_to_another_host_is_still_refused() -> None:
    requests: list[str] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_transport(
            monkeypatch,
            [(302, {"location": "http://attacker.example/health"}, b"")],
            requests,
        )
        with pytest.raises(FetchError) as error:
            fetch_url(
                "https://balangir.odisha.gov.in/health",
                source_id="downgrade-offsite",
                allowed_hosts=("balangir.odisha.gov.in",),
                resolver=_resolver,
            )
    finally:
        monkeypatch.undo()
    assert error.value.code == "scheme_not_allowed"


def test_masthead_name_does_not_hold_back_a_district_health_notice() -> None:
    """The Chief Minister masthead is chrome, so it must not trigger a hold."""

    parsed = parse_html(_PORTAL_HTML)
    assert "Majhi" in parsed.source_text
    assert "Majhi" not in parsed.body_text
    assert "dengue" in parsed.body_text

    document = Document(
        document_id="doc_masthead",
        source_id="district_khordha_health_en",
        canonical_url="https://khordha.odisha.gov.in/en/departments/health",
        retrieved_at=datetime.now(UTC),
        content_type="text/html",
        text=parsed.text,
        sha256="0" * 64,
        article_text=parsed.article_text,
    )
    signal = IngestionPipeline.default().process(document)
    assert signal.coverage_state.value == "active_direct"
    assert {match.district_id for match in signal.districts} == {"OD-DIST-khordha"}
    assert "dengue" in signal.diseases


def test_the_redactor_itself_is_untouched_by_the_chrome_carve_out() -> None:
    """Chrome changes only the hold decision; article names still hold."""

    # The masthead value is still detected and still replaced wherever it lands.
    assert redact_pii("Shri Mohan Charan Majhi").redactions[0].kind == "PERSON"

    body = _PORTAL_HTML.replace(
        b"<p>The Khordha district health office reported nine dengue cases this week.</p>",
        b"<p>Khordha: Sushanta Kumar Behera in Balipatna has tested positive for dengue.</p>",
    )
    parsed = parse_html(body)
    document = Document(
        document_id="doc_article_person",
        source_id="district_khordha_health_en",
        canonical_url="https://khordha.odisha.gov.in/en/departments/health",
        retrieved_at=datetime.now(UTC),
        content_type="text/html",
        text=parsed.text,
        sha256="1" * 64,
        article_text=parsed.article_text,
    )
    signal = IngestionPipeline.default().process(document)
    assert signal.coverage_state.value == "privacy_review_required"
    assert "Sushanta Kumar Behera" not in signal.redacted_evidence


def test_chrome_wrapper_that_owns_the_page_is_not_treated_as_chrome() -> None:
    """An unclosed layout wrapper must not delete the document it wraps."""

    parsed = parse_html(_PORTAL_HTML)
    assert parsed.warnings == ("site_chrome_removed_before_extraction",)
    assert "Vector control teams" in parsed.text
    assert "Chief Minister" not in parsed.text


def test_ocr_uses_every_installed_model_and_ignores_the_route_hint() -> None:
    languages = tesseract_languages()
    assert languages.split("+")
    assert set(languages.split("+")) <= set(OCR_LANGUAGES)
    # An English scan on an Odia-only route must not be recognised as Odia.
    assert "eng" in languages.split("+")
    # No installed model at all still yields a multi-script request, never a
    # single language taken from the route configuration.
    assert tesseract_languages(which=lambda name: None) == "+".join(OCR_LANGUAGES)


def test_link_jobs_on_an_allowlisted_alias_host_are_claimable(tmp_path: Any) -> None:
    database = Database(f"sqlite:///{tmp_path / 'alias.sqlite3'}")
    runtime = CollectionRuntime(database)
    source = runtime.registry.get("sambad_district_puri_or")
    alias = "www.sambad.in"
    assert alias in source.allowed_hosts
    assert alias not in source.url

    database.enqueue_job(
        source_id=source.id,
        kind="fetch",
        payload_ref=f"registered-link:https://{alias}/2026/07/22/puri-dengue.html",
        payload_hash="b" * 64,
        idempotency_key="alias-host-link",
    )
    claimed = runtime._claim_batch(200)
    payloads = [str(job["payload_ref"]) for job in claimed]
    assert f"registered-link:https://{alias}/2026/07/22/puri-dengue.html" in payloads


def test_status_separates_crawl_failures_from_dedup_retirements(tmp_path: Any) -> None:
    database = Database(f"sqlite:///{tmp_path / 'dead.sqlite3'}")
    runtime = CollectionRuntime(database)
    for index, (reason, url) in enumerate(
        (
            ("DUPLICATE_URL_OWNER", "https://sambad.in/a.html"),
            ("URL_CANONICALIZED", "https://sambad.in/b.html"),
            ("http_404", "https://sambad.in/c.html"),
        )
    ):
        job, _ = database.enqueue_job(
            source_id="sambad_district_puri_or",
            kind="fetch",
            payload_ref=f"registered-link:{url}",
            payload_hash=str(index) * 64,
            idempotency_key=f"dead-{index}",
        )
        with database.transaction() as connection:
            connection.execute(
                "UPDATE job SET state='dead', last_error_code=? WHERE id=?",
                (reason, job["id"]),
            )

    status = runtime.status()
    assert status["queue"]["dead"] == 3
    assert status["queue"]["dead_deduplication_retired"] == 2
    assert status["queue"]["dead_failed"] == 1
    assert status["dead_job_reasons"]["http_404"] == 1
    assert status["collection_failure_visible"] is True


def test_hindi_routes_are_district_scoped_across_most_of_odisha() -> None:
    registry = load_registry()
    enabled = [source for source in registry.sources if source.enabled]
    hindi = [source for source in enabled if "hi" in source.languages]
    scoped = {source.district_id for source in hindi if source.district_id}
    assert len(scoped) >= 25, sorted(scoped)
    known = {source.district_id for source in enabled if source.district_id}
    assert scoped <= known
    for source in hindi:
        if not source.district_id:
            continue
        state = str(source.extra.get("availability_state", ""))
        assert "collector_http_200_" in state, f"{source.id}: {state}"
