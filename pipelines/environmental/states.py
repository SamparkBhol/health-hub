from __future__ import annotations

from datetime import UTC, datetime

from .models import AcquisitionState, ProviderState


def chirps_policy_state(*, version: str, observed_at: datetime | None = None) -> ProviderState:
    if version not in {"2.0", "3.0"}:
        raise ValueError("CHIRPS version must be 2.0 or 3.0")
    return ProviderState(
        provider="UCSB Climate Hazards Center",
        product=f"CHIRPS {version} preliminary",
        state=AcquisitionState.AWAITING_SOURCE_PERMISSION_OR_APPROVED_API,
        observed_at=observed_at or datetime.now(UTC),
        reason=(
            "The selected data host robots.txt was observed as Disallow: / on "
            "2026-07-21. No automated fetch occurs until a provider-approved API, "
            "written permission, or a newly reviewed policy permits it."
        ),
        metadata={
            "robots_checked_at": "2026-07-21",
            "direct_data_host_automation": False,
            "final_must_not_replace_preliminary_vintage": True,
        },
    )


def era5_request_state(
    *,
    has_cds_credentials: bool,
    licence_accepted: bool,
    request_id: str | None = None,
    observed_at: datetime | None = None,
) -> ProviderState:
    now = observed_at or datetime.now(UTC)
    if not has_cds_credentials:
        state = AcquisitionState.CREDENTIALS_REQUIRED
        reason = "CDS credentials are not configured; no request was submitted."
    elif not licence_accepted:
        state = AcquisitionState.LICENCE_ACCEPTANCE_REQUIRED
        reason = "The dataset terms have not been accepted; no request was submitted."
    elif request_id:
        state = AcquisitionState.REQUEST_SUBMITTED
        reason = "The provider accepted the request; submission is not retrieval."
    else:
        state = AcquisitionState.NOT_REQUESTED
        reason = "Credentials and terms are ready, but no provider request id was supplied."
    return ProviderState(
        provider="ECMWF Copernicus Climate Data Store",
        product="ERA5-Land-T",
        state=state,
        observed_at=now,
        reason=reason,
        request_id=request_id,
        archive_started_no_history_before=None,
        metadata={
            "near_real_time_is_later_overwritten": True,
            "archive_start_only_after_validated_capture": True,
        },
    )
