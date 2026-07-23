from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import ssl
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from workers.ingestion.connectors import ingest_registered_url
from workers.ingestion.dedup import generate_candidates
from workers.ingestion.idsp import IdspConnector, parse_idsp_catalogue_text
from workers.ingestion.models import Document, FetchReceipt, FetchResult
from workers.ingestion.parse import (
    ParsedText,
    ParseError,
    parse_document,
    parse_html,
    tesseract_pdf_ocr,
)
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.registry import RegistryError, load_registry
from workers.ingestion.safe_fetch import FetchError, FetchPolicy, fetch_url, validate_url

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DOCUMENT_ROOT = FIXTURE_ROOT / "documents"


def _resolver(host: str, port: int, **kwargs):  # noqa: ARG001
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_document(path: Path, index: int) -> Document:
    fixture = _load_fixture(path)
    if fixture["media_type"] == "text/html":
        parsed = parse_html(fixture["body"].encode("utf-8"))
        text = parsed.text
        confidence = None
    else:
        text = fixture["ocr_text"]
        confidence = fixture.get("ocr_confidence")
    return Document(
        document_id=f"fixture-{index:02d}",
        source_id=fixture["source_id"],
        canonical_url=f"https://fixtures.invalid/{path.name}",
        retrieved_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        content_type=fixture["media_type"],
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_language_hint=fixture.get("source_language_hint"),
        ocr_confidence=confidence,
        is_synthetic_fixture=True,
    )


def test_fixture_manifest_is_complete_and_hash_pinned() -> None:
    manifest = _load_fixture(FIXTURE_ROOT / "manifest.json")
    assert len(manifest["documents"]) == 12
    for relative, expected_hash in {**manifest["documents"], **manifest["environment"]}.items():
        body = (FIXTURE_ROOT / relative).read_bytes()
        assert hashlib.sha256(body).hexdigest() == expected_hash


def test_all_twelve_fixture_contracts() -> None:
    pipeline = IngestionPipeline.default()
    paths = sorted(DOCUMENT_ROOT.glob("*.json"))
    assert len(paths) == 12
    for index, path in enumerate(paths, start=1):
        fixture = _load_fixture(path)
        expected = fixture["expected"]
        if "catalogue_rows" in expected:
            rows = parse_idsp_catalogue_text(fixture["ocr_text"])
            assert len(rows) == expected["catalogue_rows"]
            assert rows[0].outbreak_id == expected["outbreak_id"]
            assert rows[0].year == expected["year"]
            assert rows[0].week == expected["week"]
            assert rows[0].district_code == expected["district_code"]
            continue
        signal = pipeline.process(_fixture_document(path, index), as_of=date(2026, 7, 21))
        assert signal.language.value == expected["language"], path.name
        assert signal.assertion.value == expected["assertion"], path.name
        assert list(signal.diseases) == expected["diseases"], path.name
        assert [item.district_id for item in signal.districts] == expected["district_ids"], (
            path.name
        )
        assert signal.eligible_for_event_review is expected["eligible"], path.name
        if expected.get("manual_review"):
            assert signal.coverage_state.value == "language_review_required"
        for value in expected.get("contains", []):
            assert value in signal.redacted_evidence
        for value in expected.get("excludes", []):
            assert value not in signal.redacted_evidence


def test_negation_cannot_become_event() -> None:
    path = DOCUMENT_ROOT / "03_odia_negated.json"
    signal = IngestionPipeline.default().process(_fixture_document(path, 3))
    assert signal.assertion.value == "not_affirmed"
    assert not signal.eligible_for_event_review


@pytest.mark.parametrize(
    "text",
    (
        "Dengue was ruled out in Cuttack.",
        "Cuttack is free of dengue cases.",
        "There are zero dengue cases in Cuttack.",
    ),
)
def test_common_denials_are_not_event_eligible(text: str) -> None:
    document = Document(
        document_id=hashlib.sha256(text.encode()).hexdigest(),
        source_id="synthetic_test",
        canonical_url="https://fixtures.invalid/denial",
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        content_type="text/plain",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        is_synthetic_fixture=True,
    )
    signal = IngestionPipeline.default().process(document)
    assert signal.assertion.value == "not_affirmed"
    assert not signal.eligible_for_event_review


def test_common_verb_add_is_not_misclassified_as_diarrhoeal_disease() -> None:
    lexicon = IngestionPipeline.default().diseases
    assert lexicon.find("The hospital will add beds in Ganjam next week.") == ()
    assert lexicon.find("Acute diarrhoeal disease cases in Ganjam") == (
        "acute_diarrhoeal_disease",
    )


@pytest.mark.parametrize(
    "text",
    (
        "Chikungunya cases were reported in Ganjam.",
        "गंजाम में चिकनगुनिया के मामले मिले।",
        "ଗଞ୍ଜାମରେ ଚିକୁନଗୁନିଆ ମାମଲା ଚିହ୍ନଟ।",
    ),
)
def test_chikungunya_is_a_supported_phase_one_disease_group(text: str) -> None:
    assert IngestionPipeline.default().diseases.find(text) == ("chikungunya",)


def test_untitled_latin_person_pattern_is_redacted_and_privacy_held() -> None:
    text = "Ramesh Kumar in Cuttack has dengue."
    document = Document(
        document_id="ordinary-name",
        source_id="synthetic_test",
        canonical_url="https://fixtures.invalid/person",
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        content_type="text/plain",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        is_synthetic_fixture=True,
    )
    signal = IngestionPipeline.default().process(document)
    assert "Ramesh Kumar" not in signal.redacted_evidence
    assert "[PERSON_1]" in signal.redacted_evidence
    assert signal.coverage_state.value == "privacy_review_required"
    assert not signal.eligible_for_event_review


def test_cross_language_duplicates_are_review_candidates_not_merges() -> None:
    pipeline = IngestionPipeline.default()
    signals = [
        pipeline.process(_fixture_document(DOCUMENT_ROOT / "10_dedup_english.json", 10)),
        pipeline.process(_fixture_document(DOCUMENT_ROOT / "11_dedup_odia.json", 11)),
    ]
    candidates = generate_candidates(signals)
    assert len(candidates) == 1
    assert candidates[0].cross_language
    assert candidates[0].disposition == "review_required_never_auto_merge"


def test_registry_disables_unapproved_publishers() -> None:
    registry = load_registry()
    assert registry.get("idsp_weekly_outbreaks").extra["fallback"] == "wayback_cdx_then_id_endpoint"
    with pytest.raises(RegistryError):
        registry.get("dharitri")


def test_ssrf_rejects_private_resolution() -> None:
    policy = FetchPolicy.load()

    def private_resolver(host: str, port: int, **kwargs):  # noqa: ARG001
        return [(2, 1, 6, "", ("127.0.0.1", port))]

    with pytest.raises(FetchError, match="non-public") as raised:
        validate_url(
            "https://health.odisha.gov.in/en/notifications/circulars",
            allowed_hosts=("health.odisha.gov.in",),
            policy=policy,
            resolver=private_resolver,
        )
    assert raised.value.code == "non_public_address"


def test_fetch_enforces_allowlist_and_byte_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html><p>health</p></html>"

    class Response:
        status_code = 200
        headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(body)),
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def iter_bytes(self):
            yield body

    class Client:
        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def stream(self, method, url, *, headers, extensions):  # noqa: ANN001, ARG002
            return Response()

    monkeypatch.setattr("workers.ingestion.safe_fetch.httpx.Client", Client)
    result = fetch_url(
        "https://health.odisha.gov.in/test",
        source_id="test",
        allowed_hosts=("health.odisha.gov.in",),
        resolver=_resolver,
    )
    assert result.receipt.content_type == "text/html"
    assert result.receipt.sha256 == hashlib.sha256(result.body).hexdigest()
    with pytest.raises(FetchError) as raised:
        validate_url(
            "https://example.com/not-allowed",
            allowed_hosts=("health.odisha.gov.in",),
            policy=FetchPolicy.load(),
            resolver=_resolver,
        )
    assert raised.value.code == "host_not_allowlisted"


def test_fetch_accepts_json_after_response_headers_are_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"status":"ok"}'

    class Response:
        status_code = 200
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def iter_bytes(self):
            yield body

    class Client:
        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def stream(self, method, url, *, headers, extensions):  # noqa: ANN001, ARG002
            return Response()

    monkeypatch.setattr("workers.ingestion.safe_fetch.httpx.Client", Client)
    result = fetch_url(
        "https://power.larc.nasa.gov/test.json",
        source_id="nasa_power_test",
        allowed_hosts=("power.larc.nasa.gov",),
        resolver=_resolver,
    )
    assert result.receipt.content_type == "application/json"
    assert result.receipt.response_headers["content-type"].startswith(
        "application/json"
    )


def test_production_fetch_pins_validated_ip_and_preserves_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_calls: list[tuple[str, int]] = []
    client_options: list[dict[str, object]] = []
    requests: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def switching_resolver(host: str, port: int, **kwargs):  # noqa: ANN001, ARG001
        resolver_calls.append((host, port))
        # A second lookup would return loopback. The production transport must
        # connect to the public address retained from the first lookup instead.
        address = "93.184.216.34" if len(resolver_calls) == 1 else "127.0.0.1"
        return [(2, 1, 6, "", (address, port))]

    class Response:
        status_code = 200
        headers = {"content-type": "text/plain", "content-length": "2"}

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def iter_bytes(self):
            yield b"ok"

    class Client:
        def __init__(self, **kwargs):  # noqa: ANN003
            client_options.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def stream(self, method, url, *, headers, extensions):  # noqa: ANN001
            assert method == "GET"
            requests.append((url, headers, extensions))
            return Response()

    monkeypatch.setattr("workers.ingestion.safe_fetch.httpx.Client", Client)
    result = fetch_url(
        "https://health.odisha.gov.in/test?district=Khordha",
        source_id="dns-pin-test",
        allowed_hosts=("health.odisha.gov.in",),
        resolver=switching_resolver,
    )

    assert result.body == b"ok"
    assert resolver_calls == [("health.odisha.gov.in", 443)]
    assert requests == [
        (
            "https://93.184.216.34:443/test?district=Khordha",
            {
                "User-Agent": FetchPolicy.load().user_agent,
                "Accept": (
                    "text/html,application/pdf,application/json,application/xml,"
                    "text/plain;q=0.8"
                ),
                "Accept-Encoding": "identity",
                "Host": "health.odisha.gov.in",
            },
            {"sni_hostname": "health.odisha.gov.in"},
        )
    ]
    assert client_options[0]["trust_env"] is False
    assert client_options[0]["follow_redirects"] is False
    verification = client_options[0]["verify"]
    assert isinstance(verification, ssl.SSLContext)
    assert verification.check_hostname
    assert verification.verify_mode == ssl.CERT_REQUIRED


def test_production_fetch_revalidates_and_pins_each_redirect_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_calls: list[tuple[str, int]] = []
    requests: list[tuple[str, str, str]] = []
    responses = [
        (302, {"location": "https://ncdc.mohfw.gov.in/final"}, b""),
        (200, {"content-type": "text/plain", "content-length": "2"}, b"ok"),
    ]

    def resolver(host: str, port: int, **kwargs):  # noqa: ANN001, ARG001
        resolver_calls.append((host, port))
        address = {
            "health.odisha.gov.in": "93.184.216.34",
            "ncdc.mohfw.gov.in": "93.184.216.35",
        }[host]
        return [(2, 1, 6, "", (address, port))]

    class Response:
        def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.status_code = status
            self.headers = headers
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def iter_bytes(self):
            yield self.body

    class Client:
        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def stream(self, method, url, *, headers, extensions):  # noqa: ANN001
            assert method == "GET"
            requests.append((url, headers["Host"], extensions["sni_hostname"]))
            status, response_headers, body = responses.pop(0)
            return Response(status, response_headers, body)

    monkeypatch.setattr("workers.ingestion.safe_fetch.httpx.Client", Client)
    result = fetch_url(
        "https://health.odisha.gov.in/start",
        source_id="redirect-pin-test",
        allowed_hosts=(
            host for host in ("health.odisha.gov.in", "ncdc.mohfw.gov.in")
        ),
        resolver=resolver,
    )

    assert result.body == b"ok"
    assert resolver_calls == [
        ("health.odisha.gov.in", 443),
        ("ncdc.mohfw.gov.in", 443),
    ]
    assert requests == [
        (
            "https://93.184.216.34:443/start",
            "health.odisha.gov.in",
            "health.odisha.gov.in",
        ),
        (
            "https://93.184.216.35:443/final",
            "ncdc.mohfw.gov.in",
            "ncdc.mohfw.gov.in",
        ),
    ]
    assert result.receipt.redirect_chain == ("https://ncdc.mohfw.gov.in/final",)
    assert result.receipt.final_url == "https://ncdc.mohfw.gov.in/final"


def test_pdf_parser_uses_injected_ocr_and_confidence() -> None:
    def ocr_hook(body: bytes, language_hint: str | None = None) -> ParsedText:
        assert body.startswith(b"%PDF-")
        assert language_hint == "or"
        return ParsedText(text="ଖୋର୍ଦ୍ଧା ଡେଙ୍ଗୁ", ocr_confidence=0.88, parser="stub_ocr")

    parsed = parse_document(
        b"%PDF-1.4 synthetic",
        "application/pdf",
        language_hint="or",
        ocr_hook=ocr_hook,
        validate_structure=False,
    )
    assert parsed.parser == "stub_ocr"
    assert parsed.ocr_confidence == 0.88


def test_malformed_pdf_is_rejected_before_ocr() -> None:
    with pytest.raises(ParseError, match="structure"):
        parse_document(b"%PDF-1.4 not-a-real-pdf", "application/pdf")


@pytest.mark.ocr
def test_actual_ocr_binaries_process_a_raster_only_pdf() -> None:
    if shutil.which("tesseract") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Tesseract and Poppler are optional outside the production image")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 90)
    except OSError:
        pytest.skip("a deterministic TrueType test font is unavailable")
    image = Image.new("RGB", (1800, 500), "white")
    ImageDraw.Draw(image).text((80, 160), "ODISHA HEALTH ALERT", fill="black", font=font)
    payload = io.BytesIO()
    image.save(payload, format="PDF", resolution=200)

    parsed = tesseract_pdf_ocr(
        payload.getvalue(),
        "en",
        maximum_pages=1,
        timeout_seconds=45,
    )
    assert parsed.parser == "tesseract_pdf_ocr"
    assert "ODISHA" in parsed.text.upper()
    assert 0.0 <= (parsed.ocr_confidence or 0.0) <= 1.0


def test_ocr_rejects_page_count_above_lease_safe_cap() -> None:
    image = Image.new("RGB", (100, 100), "white")
    payload = io.BytesIO()
    image.save(payload, format="PDF", save_all=True, append_images=[image.copy()])
    with pytest.raises(ParseError, match="page count"):
        tesseract_pdf_ocr(
            payload.getvalue(),
            "en",
            maximum_pages=1,
            timeout_seconds=5,
        )


def test_idsp_live_failure_uses_latest_archive_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    original = "https://idsp.mohfw.gov.in/WriteReadData/l892s/example.pdf"
    calls: list[tuple[str, str]] = []
    archived_body = b"%PDF-1.4 archived"
    archived_digest = (
        base64.b32encode(hashlib.sha1(archived_body).digest())  # noqa: S324
        .decode()
        .rstrip("=")
    )

    def fake_fetch(url: str, **kwargs) -> FetchResult:
        calls.append((url, kwargs.get("access_path", "live_origin")))
        if url == original:
            raise FetchError("network_error", "origin unavailable")
        if "/cdx/search/cdx" in url:
            body = json.dumps(
                [
                    ["timestamp", "original", "statuscode", "mimetype", "digest"],
                    ["20260616010101", original, "200", "application/pdf", archived_digest],
                ]
            ).encode()
        else:
            body = archived_body
        receipt = FetchReceipt(
            source_id=kwargs["source_id"],
            requested_url=url,
            final_url=url,
            retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
            status_code=200,
            content_type="application/pdf" if body.startswith(b"%PDF") else "application/json",
            byte_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            access_path=kwargs.get("access_path", "live_origin"),
            archive_timestamp=kwargs.get("archive_timestamp"),
            archive_digest=kwargs.get("archive_digest"),
            fallback_reason=kwargs.get("fallback_reason"),
        )
        return FetchResult(receipt=receipt, body=body)

    monkeypatch.setattr("workers.ingestion.idsp.fetch_url", fake_fetch)
    result = IdspConnector().fetch_report(original)
    assert result.receipt.access_path == "wayback_id_fallback"
    assert result.receipt.archive_timestamp == "20260616010101"
    assert result.receipt.fallback_reason == "network_error"
    assert calls[0][0] == original
    assert calls[1][1] == "archive_index"
    assert calls[2][0].startswith("https://web.archive.org/web/20260616010101id_/https://idsp")


def test_idsp_parser_never_creates_negative_or_zero_weeks() -> None:
    rows = parse_idsp_catalogue_text("OR/ANU/2026/9/334 Synthetic positive catalogue row")
    assert len(rows) == 1
    assert not hasattr(rows[0], "zero_filled")


def test_archived_idsp_index_resolves_relative_report_to_canonical_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689"
    html = b'<html><a href="/WriteReadData/l892s/week9.pdf">Week 9 health PDF</a></html>'
    digest = (
        base64.b32encode(hashlib.sha1(html).digest())  # noqa: S324
        .decode()
        .rstrip("=")
    )

    def fake_fetch(url: str, **kwargs) -> FetchResult:
        if url == original:
            raise FetchError("network_error", "origin unavailable")
        if "/cdx/search/cdx" in url:
            body = json.dumps(
                [
                    ["timestamp", "original", "statuscode", "mimetype", "digest"],
                    ["20260616010101", original, "200", "text/html", digest],
                ]
            ).encode()
            media_type = "application/json"
        else:
            body = html
            media_type = "text/html"
        return FetchResult(
            receipt=FetchReceipt(
                source_id=kwargs["source_id"],
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
                status_code=200,
                content_type=media_type,
                byte_length=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                access_path=kwargs.get("access_path", "live_origin"),
                archive_timestamp=kwargs.get("archive_timestamp"),
                archive_digest=kwargs.get("archive_digest"),
                fallback_reason=kwargs.get("fallback_reason"),
            ),
            body=body,
        )

    monkeypatch.setattr("workers.ingestion.idsp.fetch_url", fake_fetch)
    outcome = ingest_registered_url(
        registry=load_registry(),
        source_id="idsp_weekly_outbreaks",
        url=original,
        pipeline=IngestionPipeline.default(),
    )
    assert outcome.receipt.requested_url == original
    assert outcome.receipt.access_path == "wayback_id_fallback"
    assert [link.url for link in outcome.discovered_links] == [
        "https://idsp.mohfw.gov.in/WriteReadData/l892s/week9.pdf"
    ]


def test_idsp_archive_fallback_rejects_body_not_matching_cdx_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "https://idsp.mohfw.gov.in/WriteReadData/l892s/tampered.pdf"

    def fake_fetch(url: str, **kwargs) -> FetchResult:
        if url == original:
            raise FetchError("network_error", "origin unavailable")
        if "/cdx/search/cdx" in url:
            body = json.dumps(
                [
                    ["timestamp", "original", "statuscode", "mimetype", "digest"],
                    ["20260616010101", original, "200", "application/pdf", "A" * 32],
                ]
            ).encode()
            content_type = "application/json"
        else:
            body = b"%PDF-1.4 bytes altered in transit"
            content_type = "application/pdf"
        return FetchResult(
            receipt=FetchReceipt(
                source_id=kwargs["source_id"],
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
                status_code=200,
                content_type=content_type,
                byte_length=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                access_path=kwargs.get("access_path", "live_origin"),
            ),
            body=body,
        )

    monkeypatch.setattr("workers.ingestion.idsp.fetch_url", fake_fetch)
    with pytest.raises(FetchError) as raised:
        IdspConnector().fetch_report(original)
    assert raised.value.code == "archive_digest_mismatch"


def test_idsp_archive_replay_requires_https_cdx_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "https://idsp.mohfw.gov.in/WriteReadData/l892s/missing-digest.pdf"

    def fake_fetch(url: str, **kwargs) -> FetchResult:
        if url == original:
            raise FetchError("network_error", "origin unavailable")
        body = json.dumps(
            [
                ["timestamp", "original", "statuscode", "mimetype", "digest"],
                ["20260616010101", original, "200", "application/pdf", None],
            ]
        ).encode()
        return FetchResult(
            receipt=FetchReceipt(
                source_id=kwargs["source_id"],
                requested_url=url,
                final_url=url,
                retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
                status_code=200,
                content_type="application/json",
                byte_length=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                access_path=kwargs.get("access_path", "live_origin"),
            ),
            body=body,
        )

    monkeypatch.setattr("workers.ingestion.idsp.fetch_url", fake_fetch)
    with pytest.raises(FetchError) as raised:
        IdspConnector().fetch_report(original)
    assert raised.value.code == "archive_digest_unavailable"
