"""Versioned response envelope.

The models intentionally reject unknown fields at domain boundaries.  Missing
scope is represented by a tagged value rather than by an ambiguous ``null``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .states import CapabilityCode, CoverageState, DeploymentProfile, LayerType

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Scoped(StrictModel, Generic[T]):
    state: Literal["value", "not_applicable", "unavailable"]
    value: T | None = None
    reason_code: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_tagged_value(self) -> Scoped[T]:
        if self.state == "value":
            if self.value is None:
                raise ValueError("value scope requires value")
            if self.reason_code is not None:
                raise ValueError("value scope cannot carry reason_code")
        elif self.value is not None:
            raise ValueError("non-value scope cannot carry value")
        if self.state == "unavailable" and not self.reason_code:
            raise ValueError("unavailable scope requires reason_code")
        if self.state == "not_applicable" and self.reason_code is not None:
            raise ValueError("not_applicable scope cannot carry reason_code")
        return self

    @classmethod
    def present(cls, value: T) -> Scoped[T]:
        return cls(state="value", value=value)

    @classmethod
    def unavailable(cls, reason_code: str, detail: str | None = None) -> Scoped[T]:
        return cls(state="unavailable", reason_code=reason_code, detail=detail)

    @classmethod
    def not_applicable(cls) -> Scoped[T]:
        return cls(state="not_applicable")


class SourceReceipt(StrictModel):
    source_id: str
    receipt_id: str
    issued_at: Scoped[datetime]
    retrieved_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataVintage(StrictModel):
    vintage_id: str
    source_receipts: list[SourceReceipt] = Field(default_factory=list)


class CapabilityState(StrictModel):
    code: CapabilityCode
    started_at: datetime | None = None

    @model_validator(mode="after")
    def require_started_at_for_archive(self) -> CapabilityState:
        if self.code == "archive_started_no_history_before" and self.started_at is None:
            raise ValueError("archive_started_no_history_before requires started_at")
        if self.code != "archive_started_no_history_before" and self.started_at is not None:
            raise ValueError("started_at is only valid for archive state")
        return self


class ProvenanceRef(StrictModel):
    source_snapshot_id: str
    canonical_url: str
    retrieved_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str
    evidence_text_version_id: str
    evidence_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coordinate_space: Literal["canonical_redacted_utf8_codepoints"] = (
        "canonical_redacted_utf8_codepoints"
    )
    evidence_offsets: list[tuple[int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_offsets(self) -> ProvenanceRef:
        if any(start < 0 or end <= start for start, end in self.evidence_offsets):
            raise ValueError("evidence offsets must be positive half-open spans")
        return self


class WarningItem(StrictModel):
    code: str
    severity: Literal["info", "warning", "blocking"]
    message: str


class DeferralItem(StrictModel):
    capability: str
    state: CapabilityState
    reason_code: str


class EnvelopeContext(StrictModel):
    layer_type: LayerType
    as_of: Scoped[datetime]
    data_vintage: Scoped[DataVintage]
    coverage_state: CoverageState
    disease_definition_version: Scoped[str]
    geography_version: Scoped[str]


class Envelope(StrictModel, Generic[T]):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    deployment_profile: DeploymentProfile = "production_shaped"
    context: EnvelopeContext
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)
    deferrals: list[DeferralItem] = Field(default_factory=list)
    data: T


class Problem(StrictModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str
    reason_code: str
    instance: str


def default_context(
    layer_type: LayerType,
    *,
    coverage_state: CoverageState = "not_applicable",
    as_of: datetime | None = None,
    data_vintage: Scoped[DataVintage] | None = None,
    disease_definition_version: Scoped[str] | None = None,
    geography_version: Scoped[str] | None = None,
) -> EnvelopeContext:
    return EnvelopeContext(
        layer_type=layer_type,
        as_of=Scoped[datetime].present(as_of or utc_now()),
        data_vintage=data_vintage or Scoped[DataVintage].not_applicable(),
        coverage_state=coverage_state,
        disease_definition_version=disease_definition_version or Scoped[str].not_applicable(),
        geography_version=geography_version or Scoped[str].not_applicable(),
    )


def dump_jsonable(model: BaseModel) -> dict[str, Any]:
    """Return JSON-ready data with RFC3339 timestamps."""

    return model.model_dump(mode="json")
