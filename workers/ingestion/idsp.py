from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass, replace

from .models import FetchResult
from .safe_fetch import FetchError, FetchPolicy, fetch_url

IDSP_HOST = "idsp.mohfw.gov.in"
WAYBACK_HOST = "web.archive.org"
CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
OUTBREAK_ID = re.compile(r"\bOR/[A-Z]{2,5}/(?P<year>20\d{2})/(?P<week>\d{1,2})/(?P<serial>\d+)\b")


@dataclass(frozen=True, slots=True)
class ArchiveCapture:
    timestamp: str
    original_url: str
    status_code: str
    media_type: str
    digest: str | None = None

    @property
    def replay_url(self) -> str:
        quoted_original = urllib.parse.quote(self.original_url, safe=":/?=&%")
        # The raw `id_` replay works over HTTPS.  Keeping the archive lookup
        # and replay on TLS removes the former cleartext exception while the
        # CDX digest still anchors the exact archived bytes.
        return f"https://{WAYBACK_HOST}/web/{self.timestamp}id_/{quoted_original}"


@dataclass(frozen=True, slots=True)
class IdspCatalogueRow:
    outbreak_id: str
    year: int
    week: int
    district_code: str
    source_text: str
    authority_status: str = "primary_official"


def _parse_cdx(body: bytes, original_url: str) -> tuple[ArchiveCapture, ...]:
    try:
        rows = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError("archive_index_invalid", "Wayback CDX response is not valid JSON") from exc
    if not isinstance(rows, list) or not rows:
        return ()
    header = rows[0]
    captures: list[ArchiveCapture] = []
    for row in rows[1:]:
        value = dict(zip(header, row, strict=False))
        if value.get("statuscode") != "200":
            continue
        captures.append(
            ArchiveCapture(
                timestamp=str(value["timestamp"]),
                original_url=str(value.get("original") or original_url),
                status_code=str(value["statuscode"]),
                media_type=str(value.get("mimetype", "application/pdf")),
                digest=value.get("digest"),
            )
        )
    return tuple(sorted(captures, key=lambda item: item.timestamp, reverse=True))


class IdspConnector:
    """Fetch IDSP reports live, then fall back to an immutable Wayback replay."""

    def __init__(self, policy: FetchPolicy | None = None) -> None:
        self.policy = policy or FetchPolicy.load()

    def discover_captures(
        self, original_url: str, *, media_type: str | None
    ) -> tuple[ArchiveCapture, ...]:
        filters = ["statuscode:200"]
        if media_type:
            filters.append(f"mimetype:{media_type}")
        query = urllib.parse.urlencode(
            {
                "url": original_url,
                "output": "json",
                "filter": filters,
                "fl": "timestamp,original,statuscode,mimetype,digest",
                "collapse": "digest",
            },
            doseq=True,
        )
        result = fetch_url(
            f"{CDX_ENDPOINT}?{query}",
            source_id="idsp_wayback_cdx",
            allowed_hosts=(WAYBACK_HOST,),
            policy=self.policy,
            access_path="archive_index",
        )
        return _parse_cdx(result.body, original_url)

    def _fetch_registered_resource(
        self, original_url: str, *, expected_media_type: str
    ) -> FetchResult:
        parsed = urllib.parse.urlsplit(original_url)
        is_report = (
            parsed.path.startswith("/WriteReadData/l892s/")
            and parsed.path.lower().endswith(".pdf")
        )
        is_index = parsed.path == "/index4.php"
        if parsed.scheme != "https" or parsed.hostname != IDSP_HOST or not (
            is_report or is_index
        ):
            raise FetchError(
                "invalid_idsp_resource_url",
                "IDSP resource must be the registered HTTPS listing or report path",
            )
        try:
            live = fetch_url(
                original_url,
                source_id="idsp_weekly_outbreaks",
                allowed_hosts=(IDSP_HOST,),
                policy=self.policy,
            )
        except FetchError as live_error:
            captures = self.discover_captures(
                original_url, media_type=expected_media_type
            )
            if not captures:
                raise FetchError(
                    "live_and_archive_unavailable",
                    f"live IDSP failed ({live_error.code}) and Wayback has no valid PDF capture",
                ) from live_error
            selected = captures[0]
            if not selected.digest or not re.fullmatch(
                r"(?:sha1:)?[A-Z2-7]{32}", selected.digest.upper()
            ):
                raise FetchError(
                    "archive_digest_unavailable",
                    "Wayback replay requires a valid SHA-1 digest from HTTPS CDX",
                ) from live_error
            archived = fetch_url(
                selected.replay_url,
                source_id="idsp_weekly_outbreaks",
                allowed_hosts=(WAYBACK_HOST,),
                policy=self.policy,
                access_path="wayback_id_fallback",
                archive_timestamp=selected.timestamp,
                archive_digest=selected.digest,
                fallback_reason=live_error.code,
            )
            # Anchor the replay bytes to the payload digest obtained from the
            # independent CDX response. Wayback CDX payload digests are base32
            # SHA-1; SHA-256 remains the application receipt digest.
            expected = selected.digest.removeprefix("sha1:").upper()
            # SHA-1 is dictated by Wayback's CDX integrity field; it is
            # verification metadata, never a password or signature.
            actual = (
                base64.b32encode(hashlib.sha1(archived.body).digest())  # noqa: S324
                .decode()
                .rstrip("=")
            )
            if actual != expected:
                raise FetchError(
                    "archive_digest_mismatch",
                    "Wayback replay bytes do not match the digest returned by HTTPS CDX",
                ) from live_error
            result = FetchResult(
                receipt=replace(archived.receipt, requested_url=original_url),
                body=archived.body,
            )
        else:
            result = live
        if result.receipt.content_type != expected_media_type:
            raise FetchError(
                "idsp_content_type_mismatch",
                f"IDSP resource is not {expected_media_type}",
            )
        return result

    def fetch_report(self, original_url: str) -> FetchResult:
        return self._fetch_registered_resource(
            original_url, expected_media_type="application/pdf"
        )

    def fetch_index(self, original_url: str) -> FetchResult:
        return self._fetch_registered_resource(original_url, expected_media_type="text/html")


def parse_idsp_catalogue_text(text: str) -> tuple[IdspCatalogueRow, ...]:
    """Extract Odisha outbreak identifiers from PDF text/OCR.

    It intentionally emits catalogue events only. It never manufactures absent
    district-weeks or interprets missing rows as zero disease.
    """

    rows: list[IdspCatalogueRow] = []
    for line in text.splitlines():
        for match in OUTBREAK_ID.finditer(line):
            outbreak_id = match.group(0)
            rows.append(
                IdspCatalogueRow(
                    outbreak_id=outbreak_id,
                    year=int(match.group("year")),
                    week=int(match.group("week")),
                    district_code=outbreak_id.split("/")[1],
                    source_text=" ".join(line.split())[:500],
                )
            )
    return tuple(rows)
