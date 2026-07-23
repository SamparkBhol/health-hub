"""Request/response payloads for mutation endpoints."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, field_validator, model_validator

from .envelope import StrictModel
from .states import ReviewDecision

LIVE_EVIDENCE_PLACEHOLDER: Final = (
    "[Live source text not retained because PII redaction is unvalidated]"
)
LIVE_EVIDENCE_REDACTION_STATE: Final[
    Literal["content_not_retained_unvalidated_pii"]
] = "content_not_retained_unvalidated_pii"


class EnqueueJobRequest(StrictModel):
    source_id: str = Field(min_length=1, max_length=100)
    kind: Literal["discover", "fetch", "parse", "replay"] = "discover"
    payload_ref: str | None = Field(default=None, max_length=500)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("payload_ref")
    @classmethod
    def prohibit_arbitrary_urls(cls, value: str | None) -> str | None:
        if value and value.lower().startswith(("http://", "https://")):
            raise ValueError("payload_ref must reference a registered object, not an arbitrary URL")
        return value


class JobClaimRequest(StrictModel):
    owner: str = Field(min_length=1, max_length=100)
    lease_seconds: int = Field(default=60, ge=15, le=900)


class RedactedSignalInput(StrictModel):
    signal_id: str | None = Field(default=None, max_length=100)
    source_id: str = Field(min_length=1, max_length=100)
    source_snapshot_id: str = Field(min_length=1, max_length=160)
    district_id: str | None = Field(default=None, max_length=100)
    disease: str | None = Field(default=None, max_length=100)
    assertion: Literal["affirmed", "not_affirmed", "speculative", "non_current"]
    evidence_text: str = Field(min_length=1, max_length=12000)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: str
    event_review_eligible: bool = False
    processing_state: Literal[
        "active_direct",
        "language_review_required",
        "privacy_review_required",
        "ambiguous_entity_linkage",
    ] = "privacy_review_required"
    redaction_state: Literal[
        "heuristic_unvalidated",
        "content_not_retained_unvalidated_pii",
    ] = "heuristic_unvalidated"
    language: Literal["or", "hi", "en", "mixed", "und"] = "und"
    extractor_version: str = Field(default="rules-v1", min_length=1, max_length=100)

    @field_validator("evidence_text")
    @classmethod
    def reject_obvious_direct_identifiers(cls, value: str) -> str:
        import re

        if re.search(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)", value):
            raise ValueError("evidence_text contains a possible Indian phone number")
        if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", value, re.I):
            raise ValueError("evidence_text contains an email address")
        return value


class SourceReceiptInput(StrictModel):
    """Non-content provenance accepted from a trusted collector."""

    source_snapshot_id: str = Field(min_length=1, max_length=160)
    source_id: str = Field(min_length=1, max_length=100)
    requested_url: str = Field(min_length=1, max_length=2000)
    final_url: str = Field(min_length=1, max_length=2000)
    retrieved_at: str
    status_code: int = Field(ge=100, le=599)
    content_type: str = Field(min_length=1, max_length=100)
    byte_length: int = Field(ge=0, le=25_000_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_path: str = Field(min_length=1, max_length=100)
    archive_timestamp: str | None = Field(default=None, max_length=32)
    archive_digest: str | None = Field(default=None, max_length=160)
    fallback_reason: str | None = Field(default=None, max_length=100)
    is_fixture: bool = False


class CompleteJobRequest(StrictModel):
    owner: str = Field(min_length=1, max_length=100)
    fencing_token: int = Field(ge=1)
    receipt: SourceReceiptInput | None = None
    signals: list[RedactedSignalInput] = Field(default_factory=list, max_length=25)


class FailJobRequest(StrictModel):
    owner: str = Field(min_length=1, max_length=100)
    fencing_token: int = Field(ge=1)
    reason_code: str = Field(min_length=1, max_length=100)
    retryable: bool


class ReviewClaimRequest(StrictModel):
    reviewer_id: str = Field(min_length=1, max_length=100)
    expected_row_version: int = Field(ge=0)
    lease_seconds: int = Field(default=900, ge=60, le=3600)


class VerifiedEventInput(StrictModel):
    district_id: str = Field(pattern=r"^OD-DIST-[a-z0-9-]+$", max_length=100)
    disease: Literal[
        "dengue",
        "malaria",
        "acute_diarrhoeal_disease",
        "cholera",
        "chikungunya",
        "aes_je",
    ]


class DecisionRequest(StrictModel):
    reviewer_id: str = Field(min_length=1, max_length=100)
    expected_row_version: int = Field(ge=0)
    decision: ReviewDecision
    rationale: str = Field(min_length=3, max_length=4000)
    supersedes_decision_id: str | None = Field(default=None, max_length=100)
    event: VerifiedEventInput | None = None

    @model_validator(mode="after")
    def validate_event_shape(self) -> DecisionRequest:
        if self.decision == "verified" and self.event is None:
            raise ValueError("verified decisions require a typed event")
        if self.decision != "verified" and self.event is not None:
            raise ValueError("only verified decisions may carry an event")
        return self


class SyntheticForecastRequest(StrictModel):
    seed: Literal[20260721] = 20260721
    horizon_weeks: Literal[1, 2, 4, 8, 12] = 1


class AgentHistoryTurn(StrictModel):
    """One bounded conversation turn supplied for follow-up resolution."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)


class AgentQueryRequest(StrictModel):
    question: str = Field(min_length=3, max_length=500)
    district_id: str | None = Field(default=None, max_length=100)
    disease: str | None = Field(default=None, max_length=100)
    maximum_evidence: int = Field(default=8, ge=1, le=20)
    #: Language the ANSWER is written in. The question may be in any of the three
    #: scripts regardless; when this is omitted the agent replies in whichever
    #: language it detected the question to be in.
    target_language: Literal["en", "hi", "or"] | None = Field(default=None)
    #: Recent turns are used only to resolve follow-ups such as "what about
    #: there?". Assistant text is never treated as evidence or indexed.
    history: list[AgentHistoryTurn] = Field(default_factory=list, max_length=8)


class TranslateRequest(StrictModel):
    """Direct translation, exposed so a reviewer can read evidence in their own script."""

    text: str = Field(min_length=1, max_length=5000)
    target_language: Literal["en", "hi", "or"]
    #: Omitted means detect from the text's script.
    source_language: Literal["en", "hi", "or"] | None = Field(default=None)
