"""Authorised Odisha district-week surveillance maps and forecast pipeline.

This module is intentionally isolated from the public-source / EpiClim paths.
It accepts only the aggregate, versioned contract in
``authorised_surveillance.py`` and supports two products:

* an observed weekly count/rate map, with completeness and data-vintage fields;
* a disease-specific probability of crossing the threshold registered in that
  same export.

There is no fill-with-zero step.  A missing district-week is missing evidence,
not a healthy district.  Forecast examples reconstruct the values available on
the issue date from ``known_at``; final vintages are used only as the future
outcome.  This is the minimum bitemporal convention needed to avoid training on
later corrections as though they were available to a historical forecaster.

The public repository does not contain an authorised export.  Consequently this
module normally returns a typed readiness/refusal response.  Once an authorised
State aggregate is placed in the documented location, ``train_and_write`` is
the reproducible, offline model build step.  No model is fitted in an API
request.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .authorised_surveillance import (
    DEFAULT_EXPORT_PATH,
    MINIMUM_CASE_VOLUME_COMPLETENESS,
    SurveillanceObservation,
    audit_export,
    load_export,
)
from .climate import FEATURE_NAMES as CLIMATE_FEATURE_NAMES
from .climate import DistrictClimateFeatures, load_weekly_panel
from .metrics import block_bootstrap, brier_score, log_score, reliability_bins, skill_score
from .models import (
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    GradientBoostedTrees,
    RidgeLogistic,
    SeasonalClimatologyBaseline,
    sigmoid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = REPO_ROOT / "data" / "forecasting" / "authorised_outbreak_model.json"
SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "authorised-district-week-threshold-v1"
FEATURE_NAMES = (
    *CLIMATE_FEATURE_NAMES,
    "target_annual_sin",
    "target_annual_cos",
    "lag_rate_1w_per_100k",
    "lag_rate_4w_per_100k",
    "lag_rate_8w_per_100k",
    "state_lag_rate_4w_per_100k",
    "case_volume_completeness_4w",
    "reporting_unit_completeness_4w",
)
SUPPORTED_HORIZONS = (1, 2, 4, 8, 12)
MINIMUM_TRAIN_WEEKS = 104
MINIMUM_EVALUATION_SEASONS = 2
MINIMUM_EVALUATION_EVENTS = 25
MAX_CURRENT_DATA_AGE_DAYS = 21


class OperationalForecastError(RuntimeError):
    """The authorised training or serving artefact is absent or invalid."""


@dataclass(frozen=True, slots=True)
class OperationalExample:
    district_id: str
    disease: str
    issue_week: date
    target_week: date
    target_week_of_year: int
    features: tuple[float, ...]
    target: int


def _week_range(start: date, end: date) -> list[date]:
    weeks: list[date] = []
    cursor = start
    while cursor <= end:
        weeks.append(cursor)
        cursor += timedelta(days=7)
    return weeks


def _latest(
    rows: Iterable[SurveillanceObservation],
) -> dict[tuple[str, str, date], SurveillanceObservation]:
    """Select the most recently known fact for each district/disease/week."""

    values: dict[tuple[str, str, date], SurveillanceObservation] = {}
    for row in rows:
        key = (row.district_id, row.disease, row.week_start)
        prior = values.get(key)
        if prior is None or row.known_at > prior.known_at:
            values[key] = row
    return values


def _as_of(
    rows: Iterable[SurveillanceObservation], *, as_of: date
) -> dict[tuple[str, str, date], SurveillanceObservation]:
    """Return the latest value that was actually known by ``as_of``."""

    values: dict[tuple[str, str, date], SurveillanceObservation] = {}
    for row in rows:
        if row.known_at > as_of:
            continue
        key = (row.district_id, row.disease, row.week_start)
        prior = values.get(key)
        if prior is None or row.known_at > prior.known_at:
            values[key] = row
    return values


def observed_surveillance_map(
    *,
    path: Path | str = DEFAULT_EXPORT_PATH,
    disease: str | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Return an official observation map without inventing values for gaps.

    The map contains the latest *available* row for every district/disease.  It
    deliberately reports both the count and rate, while the default map metric
    is rate per 100,000.  ``case_volume_completeness`` is always returned next
    to it, because it changes what a low observed count means.
    """

    rows = load_export(path)
    selected = _as_of(rows, as_of=as_of) if as_of else _latest(rows)
    by_cell: dict[tuple[str, str], SurveillanceObservation] = {}
    for row in selected.values():
        if disease and row.disease != disease:
            continue
        key = (row.district_id, row.disease)
        prior = by_cell.get(key)
        if (
            prior is None
            or row.week_start > prior.week_start
            or (row.week_start == prior.week_start and row.known_at > prior.known_at)
        ):
            by_cell[key] = row
    records = []
    for (_, _), row in sorted(
        by_cell.items(), key=lambda item: (item[1].disease, item[1].district_id)
    ):
        records.append(
            {
                "id": f"official:{row.district_id}:{row.disease}:{row.week_start}:{row.known_at}",
                "district_id": row.district_id,
                "disease": row.disease,
                "week_start": row.week_start.isoformat(),
                "known_at": row.known_at.isoformat(),
                "cases": row.cases,
                "population": row.population,
                "rate_per_100k": round(row.rate_per_100k, 6),
                "map_metric": "rate_per_100k",
                "map_value": round(row.rate_per_100k, 6),
                "case_volume_completeness": round(row.case_volume_completeness, 6),
                "reporting_unit_completeness": round(row.reporting_unit_completeness, 6),
                "reporting_units_expected": row.reporting_units_expected,
                "reporting_units_received": row.reporting_units_received,
                "case_definition_version": row.case_definition_version,
                "outbreak_threshold_per_100k": row.outbreak_threshold_per_100k,
                "threshold_version": row.threshold_version,
                "source_vintage": row.source_vintage,
                "observation_state": (
                    "observed_complete"
                    if row.case_volume_completeness >= MINIMUM_CASE_VOLUME_COMPLETENESS
                    else "observed_incomplete"
                ),
            }
        )
    return {
        "metric": "rate_per_100k",
        "metric_label": "Official observed weekly cases per 100,000 population",
        "as_of": as_of.isoformat() if as_of else None,
        "records": records,
        "rows": len(records),
        "completeness_floor": MINIMUM_CASE_VOLUME_COMPLETENESS,
        "no_data_semantics": "A district with no record is unknown, never zero incidence.",
    }


def _row_for_week(
    latest: dict[tuple[str, str, date], SurveillanceObservation],
    district_id: str,
    disease: str,
    week: date,
) -> SurveillanceObservation | None:
    return latest.get((district_id, disease, week))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _build_examples(
    rows: tuple[SurveillanceObservation, ...],
    *,
    disease: str,
    horizon_weeks: int,
    climate: dict[str, DistrictClimateFeatures],
) -> list[OperationalExample]:
    """Construct issue-time examples for one disease and horizon.

    A target is read from the final/latest vintage.  Every lag feature is
    reconstructed from rows known no later than the issue Monday.  Rows with a
    missing lag observation or low case-volume completeness are excluded; they
    are not converted to a zero or imputed value.
    """

    final = _latest(row for row in rows if row.disease == disease)
    districts = sorted({district for district, item_disease, _ in final if item_disease == disease})
    if not districts:
        return []
    all_weeks = sorted({week for _, item_disease, week in final if item_disease == disease})
    if not all_weeks:
        return []
    examples: list[OperationalExample] = []
    for issue_week in _week_range(all_weeks[0], all_weeks[-1]):
        target_week = issue_week + timedelta(weeks=horizon_weeks)
        if target_week > all_weeks[-1]:
            continue
        issue_values = _as_of((row for row in rows if row.disease == disease), as_of=issue_week)
        for district_id in districts:
            climate_features = climate.get(district_id)
            # Forecasts are issued at the start of ``issue_week``.  The latest
            # complete observed weather week is therefore the preceding week;
            # using ``issue_week`` itself would leak six future daily values.
            environment = (
                climate_features.features(issue_week - timedelta(weeks=1))
                if climate_features
                else None
            )
            if environment is None:
                # No stale value, climatological average or future observation is
                # substituted. Environment is a required feature block in this
                # model, so an uncovered issue week is not a model row.
                continue
            target_row = _row_for_week(final, district_id, disease, target_week)
            if target_row is None:
                continue
            history_weeks = [issue_week - timedelta(weeks=offset) for offset in range(1, 9)]
            history = [
                _row_for_week(issue_values, district_id, disease, week) for week in history_weeks
            ]
            if any(row is None for row in history):
                continue
            lag_rows = [row for row in history if row is not None]
            if any(
                row.case_volume_completeness < MINIMUM_CASE_VOLUME_COMPLETENESS for row in lag_rows
            ):
                continue
            state_lag_rows = [
                _row_for_week(issue_values, other, disease, issue_week - timedelta(weeks=offset))
                for other in districts
                for offset in range(1, 5)
            ]
            state_valid = [
                row
                for row in state_lag_rows
                if row is not None
                and row.case_volume_completeness >= MINIMUM_CASE_VOLUME_COMPLETENESS
            ]
            if not state_valid:
                continue
            rates = [row.rate_per_100k for row in lag_rows]
            phase = 2.0 * math.pi * (target_week.isocalendar().week - 1) / 52.1775
            features = (
                *environment,
                math.sin(phase),
                math.cos(phase),
                rates[0],
                _mean(rates[:4]),
                _mean(rates),
                _mean([row.rate_per_100k for row in state_valid]),
                _mean([row.case_volume_completeness for row in lag_rows[:4]]),
                _mean([row.reporting_unit_completeness for row in lag_rows[:4]]),
            )
            examples.append(
                OperationalExample(
                    district_id=district_id,
                    disease=disease,
                    issue_week=issue_week,
                    target_week=target_week,
                    target_week_of_year=target_week.isocalendar().week,
                    features=features,
                    target=int(target_row.rate_per_100k > target_row.outbreak_threshold_per_100k),
                )
            )
    return sorted(examples, key=lambda row: (row.issue_week, row.district_id))


def _origins(examples: list[OperationalExample]) -> list[date]:
    """One forward test season per calendar year after a two-year training window."""

    if not examples:
        return []
    first = min(row.issue_week for row in examples)
    last = max(row.issue_week for row in examples)
    origins: list[date] = []
    for year in range(first.year + 2, last.year + 1):
        candidate = date.fromisocalendar(year, 1, 1)
        if candidate > first + timedelta(weeks=MINIMUM_TRAIN_WEEKS) and candidate < last:
            origins.append(candidate)
    return origins


def _fit_predict(
    train: list[OperationalExample], test: list[OperationalExample]
) -> tuple[list[float], list[float], list[float]]:
    baseline = SeasonalClimatologyBaseline().fit(train)
    baseline_probabilities = baseline.predict(test)
    matrix = [list(row.features) for row in train]
    targets = [row.target for row in train]
    ridge = RidgeLogistic(l2=4.0).fit(matrix, targets)
    ridge_probabilities = ridge.predict([list(row.features) for row in test])
    booster = GradientBoostedTrees().fit(matrix, targets)
    booster_probabilities = booster.predict([list(row.features) for row in test])
    return baseline_probabilities, ridge_probabilities, booster_probabilities


def _evaluation(examples: list[OperationalExample]) -> dict[str, Any]:
    origins = _origins(examples)
    if len(origins) < MINIMUM_EVALUATION_SEASONS:
        return {
            "status": "insufficient_evidence",
            "reason_codes": ["FEWER_THAN_TWO_INDEPENDENT_EVALUATION_SEASONS"],
            "origins": [item.isoformat() for item in origins],
        }
    all_baseline: list[float] = []
    all_ridge: list[float] = []
    all_booster: list[float] = []
    all_target: list[int] = []
    blocks: list[tuple[list[float], list[float], list[int]]] = []
    evaluated_origins: list[dict[str, Any]] = []
    for origin in origins:
        train = [row for row in examples if row.issue_week < origin]
        # A whole ISO year is the independent evaluation block.
        test = [
            row
            for row in examples
            if origin <= row.issue_week < date.fromisocalendar(origin.isocalendar().year + 1, 1, 1)
        ]
        if len(train) < 100 or not test or len({row.target for row in train}) < 2:
            continue
        baseline, ridge, booster = _fit_predict(train, test)
        targets = [row.target for row in test]
        all_baseline.extend(baseline)
        all_ridge.extend(ridge)
        all_booster.extend(booster)
        all_target.extend(targets)
        blocks.append((ridge, baseline, targets))
        evaluated_origins.append(
            {
                "origin": origin.isoformat(),
                "rows": len(test),
                "events": sum(targets),
                "ridge_brier": round(brier_score(ridge, targets), 8),
                "baseline_brier": round(brier_score(baseline, targets), 8),
                "booster_brier": round(brier_score(booster, targets), 8),
            }
        )
    if len(blocks) < MINIMUM_EVALUATION_SEASONS or sum(all_target) < MINIMUM_EVALUATION_EVENTS:
        return {
            "status": "insufficient_evidence",
            "reason_codes": ["INSUFFICIENT_INDEPENDENT_THRESHOLD_EVENTS"],
            "origins": evaluated_origins,
            "events": sum(all_target),
        }
    ridge_brier = brier_score(all_ridge, all_target)
    baseline_brier = brier_score(all_baseline, all_target)
    booster_brier = brier_score(all_booster, all_target)
    bootstrap = block_bootstrap(blocks, replicates=500).as_dict()
    # The challenger is evaluated, but the regularised model remains the serving
    # candidate unless it demonstrably wins. This avoids a black-box promotion
    # merely because a small test sample happened to favour it.
    candidate = "ridge_logistic"
    passed = (
        ridge_brier < baseline_brier
        and log_score(all_ridge, all_target) < log_score(all_baseline, all_target)
        and float(bootstrap["delta_brier_ci_2_5"]) > 0
    )
    return {
        "status": "qualified" if passed else "insufficient_evidence",
        "reason_codes": [] if passed else ["MODEL_DID_NOT_CLEAR_CALIBRATED_BASELINE_GATE"],
        "candidate": candidate,
        "origins": evaluated_origins,
        "events": sum(all_target),
        "evaluation_rows": len(all_target),
        "ridge_brier": round(ridge_brier, 8),
        "baseline_brier": round(baseline_brier, 8),
        "booster_brier": round(booster_brier, 8),
        "ridge_log_score": round(log_score(all_ridge, all_target), 8),
        "baseline_log_score": round(log_score(all_baseline, all_target), 8),
        "ridge_brier_skill_score": round(skill_score(ridge_brier, baseline_brier), 8),
        "calibration": reliability_bins(all_ridge, all_target),
        "season_block_bootstrap": bootstrap,
    }


def _fitted_model(examples: list[OperationalExample]) -> dict[str, Any]:
    matrix = [list(row.features) for row in examples]
    targets = [row.target for row in examples]
    model = RidgeLogistic(l2=4.0).fit(matrix, targets)
    baseline = SeasonalClimatologyBaseline().fit(examples)
    return {
        "ridge": {
            "l2": model.l2,
            "coefficients": model.coefficients,
            "means": model.means,
            "deviations": model.deviations,
            "converged": model.converged,
            "iterations": model.iterations,
            "standardised_coefficients": model.standardised_coefficients(list(FEATURE_NAMES)),
        },
        "seasonal_baseline": {
            "global_rate": baseline.global_rate,
            "district_multiplier": baseline.district_multiplier,
            "week_multiplier": baseline.week_multiplier,
        },
    }


def train(
    *, path: Path | str = DEFAULT_EXPORT_PATH, horizons: tuple[int, ...] = (1, 2, 4, 8, 12)
) -> dict[str, Any]:
    """Train all eligible disease/horizon cells from an authorised export.

    This is intentionally an offline action.  It produces diagnostics even for
    cells that are refused, which gives programme owners a concrete answer to
    whether more history, a revised threshold, or a simpler baseline is needed.
    """

    audit = audit_export(path)
    if not audit.get("eligible_for_training"):
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "insufficient_evidence",
            "reason_codes": list(audit.get("reason_codes") or ["AUTHORISATION_DATA_GATE"]),
            "audit": audit,
            "results": [],
        }
    rows = load_export(path)
    try:
        climate = {
            district_id: DistrictClimateFeatures(weeks)
            for district_id, weeks in load_weekly_panel().items()
        }
    except (FileNotFoundError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "insufficient_evidence",
            "reason_codes": ["ENVIRONMENTAL_VINTAGE_ARCHIVE_UNAVAILABLE"],
            "detail": str(exc),
            "audit": audit,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    for disease in sorted({row.disease for row in rows}):
        disease_result: dict[str, Any] = {"disease": disease, "horizons": []}
        for horizon in horizons:
            if horizon not in SUPPORTED_HORIZONS:
                raise ValueError(f"unsupported horizon {horizon}")
            examples = _build_examples(
                rows, disease=disease, horizon_weeks=horizon, climate=climate
            )
            evaluation = _evaluation(examples)
            cell: dict[str, Any] = {
                "horizon_weeks": horizon,
                "examples": len(examples),
                "status": evaluation["status"],
                "evaluation": evaluation,
            }
            if evaluation["status"] == "qualified":
                cell["fitted_model"] = _fitted_model(examples)
            disease_result["horizons"].append(cell)
        results.append(disease_result)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "complete",
        "target_statement": (
            "P(official district-week rate exceeds its disease-specific, versioned threshold "
            "at horizon h | aggregate facts known on the issue date)"
        ),
        "target_is_incidence": False,
        "target_is_threshold_exceedance": True,
        "feature_names": list(FEATURE_NAMES),
        "environmental_feature_state": (
            "NASA POWER historical-vintage environmental features are required for every "
            "training and serving row; a missing climate vintage suppresses the row rather "
            "than being imputed from a later observation."
        ),
        "source_export_audit": audit,
        "results": results,
    }


def train_and_write(
    *, path: Path | str = DEFAULT_EXPORT_PATH, output: Path | str = ARTEFACT_PATH
) -> dict[str, Any]:
    report = train(path=path)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_report(path: Path | str = ARTEFACT_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise OperationalForecastError(
            f"no authorised model artefact at {source}; run scripts/train_authorised_forecast.py"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OperationalForecastError(f"invalid authorised model JSON: {exc}") from exc
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("model_version") != MODEL_VERSION
    ):
        raise OperationalForecastError("authorised model artefact schema/version is unsupported")
    return payload


def summary(path: Path | str = ARTEFACT_PATH) -> dict[str, Any]:
    report = load_report(path)
    cells = []
    for disease in report.get("results", []):
        for horizon in disease.get("horizons", []):
            cells.append(
                {
                    "disease": disease["disease"],
                    "horizon_weeks": horizon["horizon_weeks"],
                    "status": horizon["status"],
                    "reason_codes": horizon.get("evaluation", {}).get("reason_codes", []),
                    "examples": horizon.get("examples", 0),
                }
            )
    return {
        "schema_version": report["schema_version"],
        "model_version": report["model_version"],
        "status": report["status"],
        "target_statement": report.get("target_statement"),
        "target_is_threshold_exceedance": report.get("target_is_threshold_exceedance", False),
        "environmental_feature_state": report.get("environmental_feature_state"),
        "cells": cells,
    }


def _qualified_cell(report: dict[str, Any], disease: str, horizon_weeks: int) -> dict[str, Any]:
    for disease_result in report.get("results", []):
        if disease_result.get("disease") != disease:
            continue
        for cell in disease_result.get("horizons", []):
            if cell.get("horizon_weeks") == horizon_weeks:
                if cell.get("status") != "qualified":
                    raise OperationalForecastError(
                        "selected disease/horizon did not clear the calibrated baseline gate"
                    )
                return cell
    raise OperationalForecastError("no qualified disease/horizon model exists")


def _current_environment_features() -> tuple[date, dict[str, tuple[float, ...]]]:
    """Read the raw current environment feature block, not its suitability score."""

    from .current_conditions import CurrentConditionsUnavailable, current_conditions_layer

    payload = current_conditions_layer()
    features: dict[str, tuple[float, ...]] = {}
    weeks: set[date] = set()
    for district in payload.get("districts", []):
        environment = district.get("environment") or {}
        if environment.get("status") != "observed":
            continue
        week_value = environment.get("issue_week")
        raw_features = environment.get("features") or {}
        try:
            week = date.fromisoformat(str(week_value))
            vector = tuple(float(raw_features[name]) for name in CLIMATE_FEATURE_NAMES)
        except (TypeError, ValueError, KeyError):
            continue
        features[str(district["district_id"])] = vector
        weeks.add(week)
    if len(weeks) != 1 or not features:
        raise CurrentConditionsUnavailable(
            "current environmental feature layer has no single observed district-week feature block"
        )
    return next(iter(weeks)), features


def _latest_disease_week(
    final: dict[tuple[str, str, date], SurveillanceObservation], *, disease: str
) -> date | None:
    weeks_by_district: dict[str, list[date]] = defaultdict(list)
    for (district_id, item_disease, week), _ in final.items():
        if item_disease == disease:
            weeks_by_district[district_id].append(week)
    if not weeks_by_district:
        return None
    # The common edge prevents a late district from becoming the apparent
    # statewide issue date while the rest have no observation for that week.
    return min(max(weeks) for weeks in weeks_by_district.values() if weeks)


def _current_feature_rows(
    rows: tuple[SurveillanceObservation, ...],
    *,
    disease: str,
    issue_week: date,
    horizon_weeks: int,
    environment: dict[str, tuple[float, ...]],
) -> list[tuple[SurveillanceObservation, tuple[float, ...]]]:
    """Build serving vectors from values known by the environment issue week."""

    as_of_values = _as_of((row for row in rows if row.disease == disease), as_of=issue_week)
    final = _latest(row for row in rows if row.disease == disease)
    districts = sorted({district for district, item_disease, _ in final if item_disease == disease})
    output: list[tuple[SurveillanceObservation, tuple[float, ...]]] = []
    for district_id in districts:
        environment_features = environment.get(district_id)
        if environment_features is None:
            continue
        history_weeks = [issue_week - timedelta(weeks=offset) for offset in range(1, 9)]
        history = [
            _row_for_week(as_of_values, district_id, disease, week) for week in history_weeks
        ]
        if any(row is None for row in history):
            continue
        lag_rows = [row for row in history if row is not None]
        if any(row.case_volume_completeness < MINIMUM_CASE_VOLUME_COMPLETENESS for row in lag_rows):
            continue
        state_history = [
            _row_for_week(as_of_values, other, disease, issue_week - timedelta(weeks=offset))
            for other in districts
            for offset in range(1, 5)
        ]
        state_rows = [
            row
            for row in state_history
            if row is not None and row.case_volume_completeness >= MINIMUM_CASE_VOLUME_COMPLETENESS
        ]
        if not state_rows:
            continue
        reference = lag_rows[0]
        rates = [row.rate_per_100k for row in lag_rows]
        target_week = issue_week + timedelta(weeks=horizon_weeks)
        phase = 2.0 * math.pi * (target_week.isocalendar().week - 1) / 52.1775
        feature_vector = (
            *environment_features,
            math.sin(phase),
            math.cos(phase),
            rates[0],
            _mean(rates[:4]),
            _mean(rates),
            _mean([row.rate_per_100k for row in state_rows]),
            _mean([row.case_volume_completeness for row in lag_rows[:4]]),
            _mean([row.reporting_unit_completeness for row in lag_rows[:4]]),
        )
        output.append((reference, feature_vector))
    return output


def _ridge_probability(fitted: dict[str, Any], features: tuple[float, ...]) -> float:
    ridge = fitted["ridge"]
    coefficients = [float(value) for value in ridge["coefficients"]]
    means = [float(value) for value in ridge["means"]]
    deviations = [float(value) for value in ridge["deviations"]]
    if len(features) != len(means) or len(coefficients) != len(features) + 1:
        raise OperationalForecastError(
            "fitted model feature dimension does not match serving vector"
        )
    linear = coefficients[0]
    for index, value in enumerate(features):
        linear += coefficients[index + 1] * ((value - means[index]) / deviations[index])
    return min(max(sigmoid(linear), PROBABILITY_FLOOR), PROBABILITY_CEILING)


def current_probability_map(
    *,
    disease: str,
    horizon_weeks: int,
    export_path: Path | str = DEFAULT_EXPORT_PATH,
    report_path: Path | str = ARTEFACT_PATH,
) -> dict[str, Any]:
    """Score a qualified model using current climate and as-of surveillance facts.

    The call fails closed when either stream is stale, not on the same issue
    week, incomplete, or unqualified.  That is deliberate: a stale probability
    has a much worse operational meaning than an empty map.
    """

    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise OperationalForecastError("horizon must be one of 1, 2, 4, 8 or 12 weeks")
    report = load_report(report_path)
    cell = _qualified_cell(report, disease, horizon_weeks)
    rows = load_export(export_path)
    environment_week, environment = _current_environment_features()
    issue_week = environment_week + timedelta(weeks=1)
    final = _latest(row for row in rows if row.disease == disease)
    common_data_week = _latest_disease_week(final, disease=disease)
    if common_data_week is None:
        raise OperationalForecastError("requested disease is absent from authorised export")
    # The latest expected disease week must be one reporting lag behind the
    # environmental issue date.  Other alignments would combine future weather
    # with old health facts or expose a backfilled historical score as current.
    if common_data_week < issue_week - timedelta(weeks=3):
        raise OperationalForecastError(
            "authorised surveillance stream is stale for current scoring"
        )
    vectors = _current_feature_rows(
        rows,
        disease=disease,
        issue_week=issue_week,
        horizon_weeks=horizon_weeks,
        environment=environment,
    )
    if not vectors:
        raise OperationalForecastError("no complete district serving rows at current issue week")
    district_rows = []
    for reference, features in vectors:
        probability = _ridge_probability(cell["fitted_model"], features)
        district_rows.append(
            {
                "district_id": reference.district_id,
                "disease": disease,
                "issue_week": issue_week.isoformat(),
                "target_week": (issue_week + timedelta(weeks=horizon_weeks)).isoformat(),
                "horizon_weeks": horizon_weeks,
                "probability_threshold_exceedance": round(probability, 6),
                "outbreak_threshold_per_100k": reference.outbreak_threshold_per_100k,
                "threshold_version": reference.threshold_version,
                "case_definition_version": reference.case_definition_version,
                "latest_case_volume_completeness": round(reference.case_volume_completeness, 6),
                "latest_reporting_unit_completeness": round(
                    reference.reporting_unit_completeness, 6
                ),
                "source_vintage": reference.source_vintage,
                "model_version": report["model_version"],
            }
        )
    return {
        "status": "published",
        "quantity": "probability_of_disease_specific_threshold_exceedance",
        "target_statement": report["target_statement"],
        "disease": disease,
        "horizon_weeks": horizon_weeks,
        "issue_week": issue_week.isoformat(),
        "districts": sorted(district_rows, key=lambda item: item["district_id"]),
        "model_evaluation": cell["evaluation"],
        "environmental_input": (
            f"complete NASA POWER environmental feature block for {environment_week.isoformat()}"
        ),
    }
