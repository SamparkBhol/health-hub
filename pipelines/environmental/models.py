from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any


class AcquisitionState(StrEnum):
    NOT_REQUESTED = "not_requested"
    RETRIEVED_AND_VALIDATED = "retrieved_and_validated"
    FIXTURE_FALLBACK = "fixture_fallback"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AWAITING_SOURCE_PERMISSION_OR_APPROVED_API = "awaiting_source_permission_or_approved_api"
    CREDENTIALS_REQUIRED = "credentials_required"
    LICENCE_ACCEPTANCE_REQUIRED = "licence_acceptance_required"
    REQUEST_SUBMITTED = "request_submitted"
    REQUEST_QUEUED = "request_queued"
    REQUEST_FAILED = "request_failed"
    RETRIEVED_UNVALIDATED = "retrieved_unvalidated"


@dataclass(frozen=True, slots=True)
class ProviderState:
    provider: str
    product: str
    state: AcquisitionState
    observed_at: datetime
    reason: str
    request_id: str | None = None
    archive_started_no_history_before: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentalValue:
    day: date
    parameter: str
    value: float | None
    unit: str
    is_fill_value: bool = False


@dataclass(frozen=True, slots=True)
class EnvironmentalReceipt:
    provider: str
    product: str
    state: AcquisitionState
    requested_url: str
    final_url: str
    retrieved_at: datetime
    sha256: str
    byte_length: int
    longitude: float
    latitude: float
    start: date
    end: date
    api_version: str
    time_standard: str
    values: tuple[EnvironmentalValue, ...]
    warnings: tuple[str, ...]
    source_snapshot_id: str
