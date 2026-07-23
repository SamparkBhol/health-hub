from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


class AssertionClass(StrEnum):
    AFFIRMED = "affirmed"
    NOT_AFFIRMED = "not_affirmed"
    SPECULATIVE = "speculative"
    NON_CURRENT = "non_current"


class LanguageRoute(StrEnum):
    ODIA = "or"
    HINDI = "hi"
    ENGLISH = "en"
    MIXED = "mixed"
    UNDETERMINED = "und"


class CoverageState(StrEnum):
    ACTIVE_DIRECT = "active_direct"
    ACTIVE_ARCHIVE_FALLBACK = "active_archive_fallback"
    RIGHTS_PENDING = "rights_pending"
    SOURCE_UNAVAILABLE = "source_temporarily_unavailable"
    PARSER_FAILED = "parser_failed"
    LANGUAGE_REVIEW_REQUIRED = "language_review_required"
    PRIVACY_REVIEW_REQUIRED = "privacy_review_required"


@dataclass(frozen=True, slots=True)
class FetchReceipt:
    source_id: str
    requested_url: str
    final_url: str
    retrieved_at: datetime
    status_code: int
    content_type: str
    byte_length: int
    sha256: str
    access_path: str = "live_origin"
    redirect_chain: tuple[str, ...] = ()
    archive_timestamp: str | None = None
    archive_digest: str | None = None
    fallback_reason: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    # Literal `http://` Location values an origin answered with while the
    # request was made over TLS.  Retained verbatim so a downgrade is auditable
    # instead of disappearing into a generic fetch failure.
    scheme_downgrades: tuple[str, ...] = ()

    @property
    def scheme_downgraded(self) -> bool:
        return bool(self.scheme_downgrades)


@dataclass(frozen=True, slots=True)
class FetchResult:
    receipt: FetchReceipt
    body: bytes


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_id: str
    canonical_url: str
    retrieved_at: datetime
    content_type: str
    text: str
    sha256: str
    source_language_hint: str | None = None
    title: str | None = None
    ocr_confidence: float | None = None
    is_synthetic_fixture: bool = False
    # The document minus its site chrome, when the parser could separate them.
    # `text` may still hold the whole page; this names the published part.
    article_text: str | None = None

    @property
    def privacy_scan_text(self) -> str:
        """The text a personal-detail finding should be judged against.

        A masthead naming the Chief Minister appears on every page of a
        government portal.  Detecting it is correct, but treating it as a
        personal detail *disclosed by this notice* holds back the notice.
        """

        return self.article_text if self.article_text is not None else self.text


@dataclass(frozen=True, slots=True)
class DistrictMatch:
    district_id: str
    canonical_name: str
    matched_alias: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Redaction:
    kind: str
    placeholder: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    redactions: tuple[Redaction, ...]
    state: str = "heuristic_unvalidated"


@dataclass(frozen=True, slots=True)
class ExtractedSignal:
    signal_id: str
    document_id: str
    source_id: str
    canonical_url: str
    retrieved_at: datetime
    language: LanguageRoute
    assertion: AssertionClass
    diseases: tuple[str, ...]
    districts: tuple[DistrictMatch, ...]
    redacted_evidence: str
    redaction_state: str
    evidence_sha256: str
    eligible_for_event_review: bool
    coverage_state: CoverageState
    is_synthetic_fixture: bool = False
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
