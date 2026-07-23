"""Read-side access to the experimental EpiClim catalogue-row artefact.

The API layer must never re-run a backtest inside a request.  It reads the
artefact written by :mod:`packages.forecasting.backtest`, and this module is the
only place that decides what may be shown.

Two invariants are enforced here rather than in the API:

* a horizon that did not beat the seasonal climatology baseline returns a typed
  ``insufficient_evidence`` refusal and never a number;
* every payload that does return numbers carries ``is_synthetic=False``,
  ``is_incidence=False``, ``experimental=True`` and the EpiClim row-occurrence
  statement, so no client can render it as case counts, official reports or an
  operational disease forecast by accident.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from .backtest import ARTEFACT_PATH, MODEL_VERSION, NOT_INCIDENCE_WARNING, SCHEMA_VERSION
from .target import TARGET_KIND, TARGET_STATEMENT

CAPABILITY_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CAPABILITY_CATALOGUE_ONLY = "public_catalogue_only_no_denominator"


class ForecastArtefactMissing(RuntimeError):
    """No experimental EpiClim catalogue-row artefact is available."""


class ForecastArtefactInvalid(RuntimeError):
    """The artefact exists but failed its invariant checks."""


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ForecastArtefactInvalid(
            f"unexpected schema_version {payload.get('schema_version')!r}"
        )
    if payload.get("is_synthetic") is not False:
        raise ForecastArtefactInvalid("real-data artefact must declare is_synthetic=false")
    target = payload.get("target", {})
    if target.get("is_incidence") is not False or target.get("is_case_count") is not False:
        raise ForecastArtefactInvalid(
            "artefact must declare that the target is neither incidence nor a case count"
        )
    if target.get("kind") != TARGET_KIND:
        raise ForecastArtefactInvalid(
            f"artefact target must be {TARGET_KIND!r}, got {target.get('kind')!r}"
        )
    if (
        target.get("experimental") is not True
        or target.get("is_official_publication_probability") is not False
        or target.get("is_operational_forecast") is not False
    ):
        raise ForecastArtefactInvalid(
            "artefact must declare experimental=true and refuse official-publication "
            "and operational-forecast interpretations"
        )
    if not isinstance(payload.get("results"), list):
        raise ForecastArtefactInvalid("artefact carries no results list")
    return payload


def load_report(path: Path | None = None) -> dict[str, Any]:
    target = path or ARTEFACT_PATH
    if not target.exists():
        raise ForecastArtefactMissing(
            f"no real-data forecast artefact at {target}; run "
            "`python -m packages.forecasting.backtest`"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artefact
        raise ForecastArtefactInvalid(str(exc)) from exc
    return _validate(payload)


@lru_cache(maxsize=1)
def _cached_report() -> dict[str, Any]:
    return load_report()


def clear_cache() -> None:
    _cached_report.cache_clear()


def _preamble(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "model_version": report.get("model_version", MODEL_VERSION),
        "generated_at": report.get("generated_at"),
        "is_synthetic": False,
        "quantity": TARGET_KIND,
        "quantity_statement": TARGET_STATEMENT,
        "is_incidence": False,
        "is_case_count": False,
        "is_official_publication_probability": False,
        "is_operational_forecast": False,
        "experimental": True,
        "warning": NOT_INCIDENCE_WARNING,
    }


def summary(report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything a client needs to decide what it is allowed to render."""

    payload = report or _cached_report()
    cells: list[dict[str, Any]] = []
    for group in payload["results"]:
        for horizon in group["horizons"]:
            entry: dict[str, Any] = {
                "disease_group": group["disease_group"],
                "horizon_weeks": horizon["horizon_weeks"],
                "status": horizon["status"],
                "reason_codes": horizon.get("reason_codes", []),
            }
            evaluation = horizon.get("evaluation")
            if evaluation:
                entry["model_brier"] = evaluation["model_brier"]
                entry["seasonal_baseline_brier"] = evaluation["seasonal_baseline_brier"]
                entry["brier_skill_score_vs_baseline"] = evaluation["brier_skill_score_vs_baseline"]
                entry["log_score_gain_nats"] = evaluation["log_score_gain_nats"]
            cells.append(entry)
    return {
        **_preamble(payload),
        "target": payload["target"],
        "protocol": payload["protocol"],
        "models": payload["models"],
        "data": payload["data"],
        "cells": cells,
        "experimental_cells": payload.get("experimental_cells", []),
        "published_cells": [],
        "refused_cells": payload.get("refused_cells", []),
    }


def _find(report: dict[str, Any], group: str, horizon_weeks: int) -> dict[str, Any] | None:
    for entry in report["results"]:
        if entry["disease_group"] != group:
            continue
        for horizon in entry["horizons"]:
            if horizon["horizon_weeks"] == horizon_weeks:
                return horizon
    return None


def evaluation(
    group: str, horizon_weeks: int, report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the full evaluation for one cell, retained for display or not.

    Refused cells still return their diagnostics: hiding a failed evaluation
    would be worse than publishing it, as long as no probability is served.
    """

    payload = report or _cached_report()
    cell = _find(payload, group, horizon_weeks)
    if cell is None:
        return {
            **_preamble(payload),
            "disease_group": group,
            "horizon_weeks": horizon_weeks,
            "status": "insufficient_evidence",
            "capability_code": CAPABILITY_INSUFFICIENT_EVIDENCE,
            "reason_codes": ["CELL_NOT_EVALUATED"],
            "message": (
                f"No backtest exists for {group} at {horizon_weeks} weeks; nothing "
                "is available for it."
            ),
        }
    return {
        **_preamble(payload),
        "disease_group": group,
        "horizon_weeks": horizon_weeks,
        **{key: value for key, value in cell.items() if key != "latest_issue_map"},
    }


def probability_map(
    group: str, horizon_weeks: int, report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Historical EpiClim row probabilities for a retained experimental cell."""

    payload = report or _cached_report()
    cell = _find(payload, group, horizon_weeks)
    if cell is None or cell["status"] != "experimental":
        reason_codes = list(cell.get("reason_codes", [])) if cell else ["CELL_NOT_EVALUATED"]
        return {
            **_preamble(payload),
            "disease_group": group,
            "horizon_weeks": horizon_weeks,
            "status": "insufficient_evidence",
            "capability_code": CAPABILITY_INSUFFICIENT_EVIDENCE,
            "reason_codes": reason_codes,
            "districts": [],
            "message": (
                "This disease group and horizon did not beat the seasonal "
                "climatology baseline under the declared retrospective "
                "rolling-origin comparison, so no probability is served for it."
            ),
        }
    issue_map = cell.get("latest_issue_map", {})
    return {
        **_preamble(payload),
        "disease_group": group,
        "horizon_weeks": horizon_weeks,
        "status": "experimental",
        "capability_code": None,
        "reason_codes": [],
        "issue_week": issue_map.get("issue_week"),
        "target_week": issue_map.get("target_week"),
        "fitted_at_origin": issue_map.get("fitted_at_origin"),
        "support_state": issue_map.get("status", "historical_reissue"),
        "districts": issue_map.get("districts", []),
        "evaluation": cell["evaluation"],
        "calibration": cell["calibration"],
        "season_block_bootstrap": cell["season_block_bootstrap"],
        "note": issue_map.get("note"),
    }


def current_week_refusal(
    today: date | None = None, report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Refuse a present-day forecast and say exactly what would unlock it."""

    payload = report or _cached_report()
    panel_end = payload["data"]["panel"]["end"]
    return {
        **_preamble(payload),
        "status": "insufficient_evidence",
        "capability_code": CAPABILITY_CATALOGUE_ONLY,
        "reason_codes": [
            "TARGET_SERIES_ENDS_BEFORE_REQUESTED_ISSUE_DATE",
            "NO_CURRENT_AUTHORISED_OUTBREAK_REPORT_FEED",
        ],
        "asked_for": (today or date.today()).isoformat(),
        "target_series_supported_to": panel_end,
        "message": (
            "The frozen EpiClim catalogue used by the historical experiment ends "
            f"on {panel_end}, is incomplete and has no publication timestamps. Its "
            "selection process cannot be extrapolated years forward, so no EpiClim "
            "row probability or disease-risk probability is issued for the current "
            "week."
        ),
        "unlocked_by": (
            "A current, revision-aware district-week outbreak-report feed with "
            "explicit NIL weeks (for example authorised IHIP weekly reporting), "
            "which would also supply the denominators this catalogue lacks."
        ),
        "districts": [],
    }
