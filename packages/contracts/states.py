"""Closed state vocabularies shared by API, worker and UI clients."""

from typing import Literal

DeploymentProfile = Literal[
    "production_shaped",
    "enterprise_production_operational",
]

LayerType = Literal[
    "public_source_signal",
    "verified_event",
    "official_event_catalogue",
    "observed_surveillance",
    "forecast",
    # Present-day environmental favourability. Deliberately separate from
    # "forecast": it describes weather conditions now, not predicted disease.
    "environment",
    "coverage",
    "not_applicable",
]

CoverageState = Literal[
    "observed_for_registered_sources",
    "partial_registered_source_contact",
    "partial",
    "unavailable",
    "unknown",
    "not_applicable",
    "awaiting_sponsor_data",
    "source_temporarily_unavailable",
    "fixture_fallback",
]

CapabilityCode = Literal[
    "available",
    "research_only_not_operational_alert",
    "not_implemented_phase_one",
    "awaiting_sponsor_data",
    "awaiting_incumbent_comparison",
    "awaiting_boundary_licence",
    "community_demo_boundary",
    "public_catalogue_only_no_denominator",
    "translation_unavailable_source_language_only",
    "native_odia_interface_not_validated",
    "dedup_cross_language_unvalidated",
    "language_review_required",
    "privacy_review_required",
    "retention_pending_approval",
    "capacity_exceeded",
    "insufficient_evidence",
    "source_temporarily_unavailable",
    "reporting_completeness_unknown",
    "simulation_only_not_odisha_risk",
    "target_series_ineligible",
    "awaiting_source_permission_or_approved_api",
    "awaiting_external_credential",
    "archive_started_no_history_before",
]

ReviewTaskState = Literal["open", "claimed", "decided"]
ReviewDecision = Literal[
    "verified",
    "rejected",
    "needs_more_information",
    "duplicate",
]
JobState = Literal["queued", "running", "retry_wait", "completed", "dead"]
