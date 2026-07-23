"""Rolling-origin EpiClim catalogue-row experiment with a display gate.

Protocol
--------
* Origins are season boundaries (1 January).  At each origin the baseline and
  the models are refitted on every row whose *target week* falls strictly before
  the origin, then frozen.
* Evaluation uses only rows whose *issue week* is on or after the origin, so no
  evaluated forecast was ever informed by an outcome the training set had seen.
* Horizons are scored separately; nothing is pooled across horizons.
* The ridge penalty is chosen by nested rolling-origin validation *inside* the
  training window, so no evaluation season informs a hyperparameter either.
* A horizon is retained as an experimental historical map only if it beats the
  seasonal climatology baseline on
  BOTH strictly proper scores (Brier and logarithmic), and both advantages
  survive a 95 percent season-block bootstrap.  Otherwise the horizon returns
  ``insufficient_evidence`` and no number at all.
* An environment ablation is fitted alongside every cell, so the artefact can
  say whether the environmental block earned its place.

The quantity being scored is whether the frozen, incomplete EpiClim file has a
matching row dated in a district-week. It is not incidence, an official report-
publication probability, or an operational outbreak forecast. Because the file
has no publication timestamps, the rolling-origin design limits ordinary target
leakage but cannot reconstruct true historical knowledge time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pipelines.environmental.districts import POINT_WARNINGS, load_district_points
from pipelines.environmental.historical import read_manifest

from . import metrics
from .climate import FEATURE_NAMES as CLIMATE_FEATURE_NAMES
from .climate import (
    FEATURE_NOTES,
    MIN_PRIOR_YEARS,
    build_feature_index,
    load_weekly_panel,
)
from .models import GradientBoostedTrees, RidgeLogistic, SeasonalClimatologyBaseline, logit
from .panel import (
    DEFAULT_REPORT_LAG_WEEKS,
    HISTORY_FEATURE_NAMES,
    PANEL_FEATURE_NAMES,
    SEASONAL_FEATURE_NAMES,
    Example,
    build_examples,
    panel_weeks,
)
from .target import (
    DISEASE_GROUPS,
    EPICLIM_STRING_OVERLAY,
    TARGET_KIND,
    TARGET_STATEMENT,
    build_target_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = REPO_ROOT / "data" / "forecasting" / "reported_outbreak_model.json"
SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "experimental-epiclim-catalogue-row-1.0.0"

DEFAULT_HORIZONS = (1, 2, 4, 8, 12)
DEFAULT_EVALUATION_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
# The week grid starts at the first year of the outbreak catalogue so that
# reporting-history counters accumulate from 2009, even though modelling rows
# only begin once the environmental anomaly climatology has enough prior years.
PANEL_START = date(2009, 1, 1)
PANEL_END = date(2022, 12, 31)

MINIMUM_EVALUATION_EVENTS = 25
MINIMUM_TRAINING_EVENTS = 40

L2_GRID = (0.5, 2.0, 8.0, 32.0, 128.0)
INNER_VALIDATION_SEASONS = 2

# Column blocks of PANEL_FEATURE_NAMES; the baseline log-odds column is always
# appended last so every variant is anchored on the same climatology.
_ENVIRONMENT_WIDTH = len(CLIMATE_FEATURE_NAMES)
_CALENDAR_WIDTH = len(SEASONAL_FEATURE_NAMES)
_HISTORY_WIDTH = len(HISTORY_FEATURE_NAMES)
FEATURE_BLOCKS: dict[str, tuple[int, int]] = {
    "environment": (0, _ENVIRONMENT_WIDTH),
    "calendar": (_ENVIRONMENT_WIDTH, _ENVIRONMENT_WIDTH + _CALENDAR_WIDTH),
    "reporting_history": (
        _ENVIRONMENT_WIDTH + _CALENDAR_WIDTH,
        _ENVIRONMENT_WIDTH + _CALENDAR_WIDTH + _HISTORY_WIDTH,
    ),
}
VARIANTS: dict[str, tuple[str, ...]] = {
    "environment_and_reporting_history": ("environment", "calendar", "reporting_history"),
    "reporting_history_only": ("calendar", "reporting_history"),
}
PRIMARY_VARIANT = "environment_and_reporting_history"
ABLATION_VARIANT = "reporting_history_only"

NOT_INCIDENCE_WARNING = (
    "Experimental historical result only: this is the probability that the "
    "frozen, incomplete EpiClim file contains a matching row dated in a "
    "district-week. It is not incidence, not a case count, not a probability of "
    "official publication and not an operational outbreak forecast. A zero means "
    "only that this file contains no matching row."
)


@dataclass(frozen=True, slots=True)
class OriginResult:
    year: int
    horizon_weeks: int
    train_rows: int
    train_events: int
    test_rows: int
    test_events: int
    model_brier: float
    baseline_brier: float
    challenger_brier: float | None
    model_log_score: float
    baseline_log_score: float


def variant_columns(variant: str) -> list[int]:
    blocks = VARIANTS[variant]
    columns: list[int] = []
    for block in blocks:
        start, end = FEATURE_BLOCKS[block]
        columns.extend(range(start, end))
    return columns


def variant_feature_names(variant: str) -> list[str]:
    return [PANEL_FEATURE_NAMES[index] for index in variant_columns(variant)] + [
        "seasonal_baseline_logit"
    ]


def _design(rows: list[Example], baseline: list[float], columns: list[int]) -> list[list[float]]:
    return [
        [*(row.features[index] for index in columns), logit(probability)]
        for row, probability in zip(rows, baseline, strict=True)
    ]


def _split(rows: list[Example], origin: date, year: int) -> tuple[list[Example], list[Example]]:
    train = [row for row in rows if row.target_week < origin]
    test = [row for row in rows if row.issue_week >= origin and row.target_week.year == year]
    return train, test


def select_l2(
    train: list[Example],
    columns: list[int],
    *,
    grid: tuple[float, ...] = L2_GRID,
    folds: int = INNER_VALIDATION_SEASONS,
    default: float = 8.0,
) -> tuple[float, list[dict[str, object]]]:
    """Choose the ridge penalty inside the training window only.

    The evaluation seasons are never touched: inner validation re-uses the same
    rolling-origin rule one level down, so the reported skill is not the skill of
    a hyperparameter tuned on the data it is scored against.
    """

    years = sorted({row.target_week.year for row in train})
    inner_years = years[-folds:] if len(years) > folds else years[1:]
    trace: list[dict[str, object]] = []
    totals = {value: 0.0 for value in grid}
    usable = 0
    for year in inner_years:
        inner_origin = date(year, 1, 1)
        inner_train, inner_valid = _split(train, inner_origin, year)
        if not inner_train or not inner_valid:
            continue
        if sum(row.target for row in inner_train) < MINIMUM_TRAINING_EVENTS // 2:
            continue
        if sum(row.target for row in inner_valid) < 1:
            continue
        usable += 1
        baseline = SeasonalClimatologyBaseline().fit(inner_train)
        design_train = _design(inner_train, baseline.predict(inner_train), columns)
        design_valid = _design(inner_valid, baseline.predict(inner_valid), columns)
        targets_train = [row.target for row in inner_train]
        targets_valid = [row.target for row in inner_valid]
        fold: dict[str, float] = {}
        for value in grid:
            model = RidgeLogistic(l2=value).fit(design_train, targets_train)
            score = metrics.log_score(model.predict(design_valid), targets_valid)
            totals[value] += score * len(targets_valid)
            fold[str(value)] = round(score, 6)
        trace.append({"inner_season": year, "validation_log_score": fold})
    if usable == 0:
        return default, trace
    best = min(grid, key=lambda value: totals[value])
    return best, trace


def _variant_summary(
    name: str,
    model_probabilities: list[float],
    baseline_probabilities: list[float],
    targets: list[int],
) -> dict[str, object]:
    model_brier = metrics.brier_score(model_probabilities, targets)
    baseline_brier = metrics.brier_score(baseline_probabilities, targets)
    model_log = metrics.log_score(model_probabilities, targets)
    baseline_log = metrics.log_score(baseline_probabilities, targets)
    return {
        "variant": name,
        "brier": round(model_brier, 8),
        "log_score": round(model_log, 6),
        "brier_skill_score_vs_baseline": round(metrics.skill_score(model_brier, baseline_brier), 6),
        "log_score_gain_nats": round(baseline_log - model_log, 6),
        "auc": (
            round(value, 4)
            if (value := metrics.auc(model_probabilities, targets)) is not None
            else None
        ),
    }


def evaluate_horizon(
    rows: list[Example],
    *,
    horizon_weeks: int,
    evaluation_years: tuple[int, ...],
    challenger: bool,
    l2: float | None,
    seed: int,
) -> dict[str, object]:
    """Roll forward one season at a time and score every rung of the ladder."""

    origin_results: list[OriginResult] = []
    season_blocks: list[tuple[list[float], list[float], list[int]]] = []
    pooled: dict[str, list[float]] = {name: [] for name in VARIANTS}
    pooled_baseline: list[float] = []
    pooled_challenger: list[float] = []
    pooled_targets: list[int] = []
    coefficients: dict[str, float] = {}
    chosen_l2: dict[str, float] = {}
    tuning_trace: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for year in evaluation_years:
        origin = date(year, 1, 1)
        train, test = _split(rows, origin, year)
        if not train or not test:
            skipped.append({"year": year, "reason_code": "EMPTY_TRAIN_OR_TEST_SPLIT"})
            continue
        train_events = sum(row.target for row in train)
        if train_events < MINIMUM_TRAINING_EVENTS:
            skipped.append(
                {
                    "year": year,
                    "reason_code": "TRAINING_EVENTS_BELOW_MINIMUM",
                    "training_events": train_events,
                    "minimum": MINIMUM_TRAINING_EVENTS,
                }
            )
            continue
        baseline_model = SeasonalClimatologyBaseline().fit(train)
        baseline_train = baseline_model.predict(train)
        baseline_test = baseline_model.predict(test)
        targets_train = [row.target for row in train]
        targets_test = [row.target for row in test]

        predictions: dict[str, list[float]] = {}
        for name in VARIANTS:
            columns = variant_columns(name)
            if l2 is None:
                penalty, folds = select_l2(train, columns)
                tuning_trace.append(
                    {
                        "origin": origin.isoformat(),
                        "variant": name,
                        "selected_l2": penalty,
                        "inner_validation": folds,
                    }
                )
            else:
                penalty = l2
            chosen_l2[name] = penalty
            design_train = _design(train, baseline_train, columns)
            design_test = _design(test, baseline_test, columns)
            model = RidgeLogistic(l2=penalty).fit(design_train, targets_train)
            predictions[name] = model.predict(design_test)
            if name == PRIMARY_VARIANT:
                coefficients = model.standardised_coefficients(variant_feature_names(name))

        challenger_test: list[float] | None = None
        if challenger:
            columns = variant_columns(PRIMARY_VARIANT)
            booster = GradientBoostedTrees().fit(
                _design(train, baseline_train, columns), targets_train
            )
            challenger_test = booster.predict(_design(test, baseline_test, columns))

        primary = predictions[PRIMARY_VARIANT]
        origin_results.append(
            OriginResult(
                year=year,
                horizon_weeks=horizon_weeks,
                train_rows=len(train),
                train_events=train_events,
                test_rows=len(test),
                test_events=sum(targets_test),
                model_brier=metrics.brier_score(primary, targets_test),
                baseline_brier=metrics.brier_score(baseline_test, targets_test),
                challenger_brier=(
                    metrics.brier_score(challenger_test, targets_test)
                    if challenger_test is not None
                    else None
                ),
                model_log_score=metrics.log_score(primary, targets_test),
                baseline_log_score=metrics.log_score(baseline_test, targets_test),
            )
        )
        season_blocks.append((primary, baseline_test, targets_test))
        for name, values in predictions.items():
            pooled[name].extend(values)
        pooled_baseline.extend(baseline_test)
        pooled_targets.extend(targets_test)
        if challenger_test is not None:
            pooled_challenger.extend(challenger_test)

    if not origin_results:
        return {
            "horizon_weeks": horizon_weeks,
            "status": "insufficient_evidence",
            "reason_codes": ["NO_USABLE_ROLLING_ORIGIN"],
            "skipped_origins": skipped,
        }

    pooled_model = pooled[PRIMARY_VARIANT]
    pooled_events = sum(pooled_targets)
    model_brier = metrics.brier_score(pooled_model, pooled_targets)
    baseline_brier = metrics.brier_score(pooled_baseline, pooled_targets)
    model_log = metrics.log_score(pooled_model, pooled_targets)
    baseline_log = metrics.log_score(pooled_baseline, pooled_targets)
    bootstrap = metrics.block_bootstrap(season_blocks, seed=seed)

    reason_codes: list[str] = []
    if pooled_events < MINIMUM_EVALUATION_EVENTS:
        reason_codes.append("EVALUATION_EVENTS_BELOW_MINIMUM")
    if model_brier >= baseline_brier:
        reason_codes.append("DOES_NOT_BEAT_SEASONAL_CLIMATOLOGY_BRIER")
    if model_log >= baseline_log:
        reason_codes.append("DOES_NOT_BEAT_SEASONAL_CLIMATOLOGY_LOG_SCORE")
    if bootstrap.lower_delta_brier <= 0:
        reason_codes.append("BRIER_SEASON_BOOTSTRAP_INTERVAL_INCLUDES_ZERO")
    if bootstrap.lower_delta_log <= 0:
        reason_codes.append("LOG_SCORE_SEASON_BOOTSTRAP_INTERVAL_INCLUDES_ZERO")

    ablation = _variant_summary(
        ABLATION_VARIANT, pooled[ABLATION_VARIANT], pooled_baseline, pooled_targets
    )
    ablation_brier = float(ablation["brier"])  # type: ignore[arg-type]
    ablation["environment_brier_gain"] = round(ablation_brier - model_brier, 8)
    ablation["environment_log_score_gain_nats"] = round(
        metrics.log_score(pooled[ABLATION_VARIANT], pooled_targets) - model_log, 6
    )
    ablation["interpretation"] = (
        "Positive gains mean the environmental block adds information beyond "
        "calendar and reporting history; a value at or below zero means the "
        "environment contributed nothing measurable at this horizon."
    )

    payload: dict[str, object] = {
        "horizon_weeks": horizon_weeks,
        "status": "experimental" if not reason_codes else "insufficient_evidence",
        "reason_codes": reason_codes,
        "selected_l2_last_origin": chosen_l2,
        "evaluation": {
            "rows": len(pooled_targets),
            "events": pooled_events,
            "event_rate": round(pooled_events / len(pooled_targets), 6),
            "model_brier": round(model_brier, 8),
            "seasonal_baseline_brier": round(baseline_brier, 8),
            "brier_skill_score_vs_baseline": round(
                metrics.skill_score(model_brier, baseline_brier), 6
            ),
            "model_log_score": round(model_log, 6),
            "seasonal_baseline_log_score": round(baseline_log, 6),
            "log_score_gain_nats": round(baseline_log - model_log, 6),
            "model_auc": (
                round(value, 4)
                if (value := metrics.auc(pooled_model, pooled_targets)) is not None
                else None
            ),
            "seasonal_baseline_auc": (
                round(value, 4)
                if (value := metrics.auc(pooled_baseline, pooled_targets)) is not None
                else None
            ),
        },
        "environment_ablation": ablation,
        "calibration": {
            "reliability": metrics.reliability_bins(pooled_model, pooled_targets),
            "baseline_reliability": metrics.reliability_bins(pooled_baseline, pooled_targets),
            "expected_calibration_error": metrics.expected_calibration_error(
                pooled_model, pooled_targets
            ),
            "baseline_expected_calibration_error": metrics.expected_calibration_error(
                pooled_baseline, pooled_targets
            ),
            "randomised_pit": metrics.randomised_pit(pooled_model, pooled_targets, seed=seed),
        },
        "season_block_bootstrap": bootstrap.as_dict(),
        "rolling_origins": [
            {
                "origin": f"{item.year}-01-01",
                "evaluation_season": item.year,
                "train_rows": item.train_rows,
                "train_events": item.train_events,
                "test_rows": item.test_rows,
                "test_events": item.test_events,
                "model_brier": round(item.model_brier, 8),
                "seasonal_baseline_brier": round(item.baseline_brier, 8),
                "brier_skill_score": round(
                    metrics.skill_score(item.model_brier, item.baseline_brier), 6
                ),
                "model_log_score": round(item.model_log_score, 6),
                "seasonal_baseline_log_score": round(item.baseline_log_score, 6),
                "challenger_brier": (
                    round(item.challenger_brier, 8) if item.challenger_brier is not None else None
                ),
            }
            for item in origin_results
        ],
        "skipped_origins": skipped,
        "hyperparameter_selection": tuning_trace,
        "standardised_coefficients_last_origin": coefficients,
    }
    if pooled_challenger:
        challenger_brier = metrics.brier_score(pooled_challenger, pooled_targets)
        payload["challenger"] = {
            "name": "gradient_boosted_trees",
            "brier": round(challenger_brier, 8),
            "log_score": round(metrics.log_score(pooled_challenger, pooled_targets), 6),
            "brier_skill_score_vs_baseline": round(
                metrics.skill_score(challenger_brier, baseline_brier), 6
            ),
            "brier_skill_score_vs_logistic": round(
                metrics.skill_score(challenger_brier, model_brier), 6
            ),
            "auc": (
                round(value, 4)
                if (value := metrics.auc(pooled_challenger, pooled_targets)) is not None
                else None
            ),
            "promoted": False,
            "promotion_note": (
                "The challenger is reported for transparency only. The experimental "
                "path is the regularised logistic model; promoting a booster on "
                "this event count would be over-fitting a few hundred reports."
            ),
        }
    return payload


def latest_issue_map(
    rows: list[Example],
    *,
    horizon_weeks: int,
    origin_year: int,
    l2: float | None,
) -> dict[str, object]:
    """Fit at the last origin and score the last fully supported issue week."""

    origin = date(origin_year, 1, 1)
    train = [row for row in rows if row.target_week < origin]
    candidates = [row for row in rows if row.issue_week >= origin]
    if not train or not candidates:
        return {"status": "unavailable", "reason_code": "NO_SUPPORTED_ISSUE_WEEK"}
    last_issue = max(row.issue_week for row in candidates)
    selected = [row for row in candidates if row.issue_week == last_issue]
    columns = variant_columns(PRIMARY_VARIANT)
    baseline_model = SeasonalClimatologyBaseline().fit(train)
    baseline_train = baseline_model.predict(train)
    baseline_selected = baseline_model.predict(selected)
    penalty = l2 if l2 is not None else select_l2(train, columns)[0]
    logistic = RidgeLogistic(l2=penalty).fit(
        _design(train, baseline_train, columns), [row.target for row in train]
    )
    probabilities = logistic.predict(_design(selected, baseline_selected, columns))
    names = {point.district_id: point.canonical_name for point in load_district_points()}
    return {
        "status": "historical_reissue",
        "issue_week": last_issue.isoformat(),
        "target_week": selected[0].target_week.isoformat(),
        "fitted_at_origin": origin.isoformat(),
        "quantity": TARGET_KIND,
        "districts": [
            {
                "district_id": row.district_id,
                "canonical_name": names.get(row.district_id, row.district_id),
                "probability_epiclim_catalogue_row": round(probability, 6),
                "seasonal_baseline_probability": round(base, 6),
                "observed_epiclim_catalogue_row": row.target,
            }
            for row, probability, base in sorted(
                zip(selected, probabilities, baseline_selected, strict=True),
                key=lambda item: item[0].district_id,
            )
        ],
        "note": (
            "Out-of-sample reissue at the last issue week the target series "
            "supports. It is a retrospective probability of membership in the "
            "frozen EpiClim file, not an official-publication probability, current "
            "outbreak risk or incidence."
        ),
    }


def run_backtest(
    *,
    groups: tuple[str, ...] = ("any_reported_outbreak", "diarrhoeal_and_cholera", "vector_borne"),
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    evaluation_years: tuple[int, ...] = DEFAULT_EVALUATION_YEARS,
    report_lag_weeks: int = DEFAULT_REPORT_LAG_WEEKS,
    challenger: bool = False,
    l2: float | None = None,
    seed: int = 20260721,
    progress=None,
) -> dict[str, object]:
    weekly = load_weekly_panel()
    climate = build_feature_index(weekly)
    weeks = panel_weeks(PANEL_START, PANEL_END)
    manifest = read_manifest()
    points = {point.district_id: point for point in load_district_points()}
    reference_panel = build_target_panel("any_reported_outbreak")
    first_issue_week = min(
        (
            week.isoformat()
            for week in weeks
            for features in climate.values()
            if features.features(week) is not None
        ),
        default=None,
    )

    results: list[dict[str, Any]] = []
    for group in groups:
        panel = build_target_panel(group)
        horizon_payloads: list[dict[str, Any]] = []
        group_payload: dict[str, Any] = {
            "disease_group": group,
            "diseases": list(DISEASE_GROUPS[group]),
            "catalogue_rows": len(panel.events),
            "distinct_district_weeks": panel.positive_count,
            "horizons": horizon_payloads,
        }
        if panel.positive_count < MINIMUM_EVALUATION_EVENTS:
            group_payload["status"] = "insufficient_evidence"
            group_payload["reason_codes"] = ["CATALOGUE_EVENTS_BELOW_MINIMUM"]
            group_payload["detail"] = (
                f"{panel.positive_count} distinct district-weeks in the whole "
                f"catalogue is below the {MINIMUM_EVALUATION_EVENTS}-event floor "
                "required before any horizon is even fitted."
            )
            horizon_payloads.extend(
                {
                    "horizon_weeks": horizon,
                    "status": "insufficient_evidence",
                    "reason_codes": ["CATALOGUE_EVENTS_BELOW_MINIMUM"],
                }
                for horizon in horizons
            )
            results.append(group_payload)
            if progress:
                progress(group, 0, "refused_before_fit")
            continue
        group_payload["status"] = "evaluated"
        for horizon in horizons:
            rows = build_examples(
                target_panel=panel,
                climate=climate,
                horizon_weeks=horizon,
                weeks=weeks,
                report_lag_weeks=report_lag_weeks,
            )
            payload = evaluate_horizon(
                rows,
                horizon_weeks=horizon,
                evaluation_years=evaluation_years,
                challenger=challenger,
                l2=l2,
                seed=seed,
            )
            if payload["status"] == "experimental":
                payload["latest_issue_map"] = latest_issue_map(
                    rows,
                    horizon_weeks=horizon,
                    origin_year=evaluation_years[-1],
                    l2=l2,
                )
            horizon_payloads.append(payload)
            if progress:
                progress(group, horizon, str(payload["status"]))
        results.append(group_payload)

    experimental = [
        {"disease_group": group["disease_group"], "horizon_weeks": item["horizon_weeks"]}
        for group in results
        for item in group["horizons"]
        if item["status"] == "experimental"
    ]
    refused = [
        {
            "disease_group": group["disease_group"],
            "horizon_weeks": item["horizon_weeks"],
            "reason_codes": item.get("reason_codes", []),
        }
        for group in results
        for item in group["horizons"]
        if item["status"] != "experimental"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "is_synthetic": False,
        "uses_real_odisha_data": True,
        "target": {
            "kind": TARGET_KIND,
            "statement": TARGET_STATEMENT,
            "is_incidence": False,
            "is_case_count": False,
            "is_official_publication_probability": False,
            "is_operational_forecast": False,
            "experimental": True,
            "zero_means": "no matching district-week row in this frozen EpiClim file",
            "index_uncertainty": (
                "The catalogue's own week label disagrees with its row date by "
                "more than one ISO week for 28 percent of national rows and some "
                "discrepancies are much larger. Rows are indexed by their parsed "
                "date as a convention, not as a publication timestamp."
            ),
        },
        "protocol": {
            "design": "expanding-window rolling origin, annual refit at each season boundary",
            "training_rule": "every row whose target week falls strictly before the origin",
            "evaluation_rule": "rows whose issue week is on or after the origin",
            "random_splits_used": False,
            "report_lag_weeks": report_lag_weeks,
            "report_lag_rationale": (
                "Sensitivity assumption only: EpiClim has no publication timestamp, "
                "so row history is truncated at issue week minus this fixed lag. "
                "This does not prove historical availability."
            ),
            "knowledge_time_reconstructed": False,
            "horizons_scored_separately": True,
            "hyperparameter_selection": (
                "fixed"
                if l2 is not None
                else "nested rolling-origin validation inside the training window only"
            ),
            "experimental_display_gate": {
                "minimum_evaluation_events": MINIMUM_EVALUATION_EVENTS,
                "minimum_training_events": MINIMUM_TRAINING_EVENTS,
                "must_beat_baseline_brier": True,
                "must_beat_baseline_log_score": True,
                "brier_season_bootstrap_interval_must_exclude_zero": True,
                "log_score_season_bootstrap_interval_must_exclude_zero": True,
                "bootstrap_interval": "95 percent, resampling whole evaluation seasons",
            },
        },
        "models": {
            "baseline": "seasonal_climatology_district_x_smoothed_week_of_year",
            "experimental": "l2_logistic_regression_irls",
            "ablation": "l2_logistic_regression_without_environmental_block",
            "challenger": "gradient_boosted_trees" if challenger else "not_run",
            "l2": "selected_per_origin" if l2 is None else l2,
            "l2_grid": list(L2_GRID),
        },
        "features": {
            "names": [*PANEL_FEATURE_NAMES, "seasonal_baseline_logit"],
            "blocks": {name: list(bounds) for name, bounds in FEATURE_BLOCKS.items()},
            "variants": {name: list(blocks) for name, blocks in VARIANTS.items()},
            "notes": FEATURE_NOTES,
        },
        "data": {
            "environment": {
                "provider": "NASA POWER",
                "product": "daily_point_v2",
                "districts": len(manifest),
                "window": (
                    f"{min(item.start for item in manifest.values()).isoformat()}"
                    f"..{max(item.end for item in manifest.values()).isoformat()}"
                )
                if manifest
                else None,
                "point_selection": sorted(
                    {points[key].method for key in manifest if key in points}
                ),
                "warnings": list(POINT_WARNINGS),
            },
            "target_catalogue": {
                "name": "EpiClim (Zenodo 14580510)",
                "sha256": reference_panel.dataset_sha256,
                "authority_status": "secondary_derived",
                "positive_only": True,
                "nil_weeks_present": False,
                "denominator_present": False,
                "publication_timestamp_present": False,
                "completeness_known": False,
                "row_resolution": reference_panel.resolution,
                "district_string_overlay": {
                    raw: {"district_id": district_id, "reason": reason}
                    for raw, (district_id, reason) in EPICLIM_STRING_OVERLAY.items()
                },
            },
            "panel": {
                "start": PANEL_START.isoformat(),
                "end": PANEL_END.isoformat(),
                "districts": len(climate),
                "week_count": len(weeks),
                "first_modelled_issue_week": first_issue_week,
                "anomaly_minimum_prior_years": MIN_PRIOR_YEARS,
                "note": (
                    "The week grid starts in 2009 so reporting history accumulates "
                    "from the first catalogue year, but modelling rows only begin "
                    "once the anomaly climatology has enough prior years."
                ),
            },
        },
        "results": results,
        "experimental_cells": experimental,
        "published_cells": [],
        "refused_cells": refused,
        "warnings": [NOT_INCIDENCE_WARNING],
    }


def write_report(report: dict[str, object], path: Path | None = None) -> Path:
    target = path or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _summary_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for group in report["results"]:
        lines.append(f"[{group['disease_group']}] status={group['status']}")
        for item in group["horizons"]:
            if item["status"] != "experimental":
                lines.append(
                    f"  h={item['horizon_weeks']:>2}w  REFUSED  "
                    f"{','.join(item.get('reason_codes', [])) or 'no_reason'}"
                )
                continue
            evaluation = item["evaluation"]
            bootstrap = item["season_block_bootstrap"]
            lines.append(
                f"  h={item['horizon_weeks']:>2}w  EXPERIMENTAL "
                f"brier={evaluation['model_brier']:.6f} "
                f"base={evaluation['seasonal_baseline_brier']:.6f} "
                f"bss={evaluation['brier_skill_score_vs_baseline']:+.4f} "
                f"logloss={evaluation['model_log_score']:.5f} "
                f"base_logloss={evaluation['seasonal_baseline_log_score']:.5f} "
                f"ci=[{bootstrap['delta_brier_ci_2_5']:+.2e},"
                f"{bootstrap['delta_brier_ci_97_5']:+.2e}]"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", action="store_true")
    parser.add_argument(
        "--l2",
        type=float,
        default=None,
        help="fix the ridge penalty; omit to select it by nested rolling-origin validation",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--report-lag-weeks", type=int, default=DEFAULT_REPORT_LAG_WEEKS)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--groups", nargs="+", default=list(DISEASE_GROUPS))
    parser.add_argument("--output", type=Path, default=ARTEFACT_PATH)
    args = parser.parse_args(argv)

    def progress(group: str, horizon: int, status: str) -> None:
        print(f"  ... {group} h={horizon} -> {status}", file=sys.stderr, flush=True)

    report = run_backtest(
        groups=tuple(args.groups),
        horizons=tuple(args.horizons),
        report_lag_weeks=args.report_lag_weeks,
        challenger=args.challenger,
        l2=args.l2,
        seed=args.seed,
        progress=progress,
    )
    path = write_report(report, args.output)
    print("\n".join(_summary_lines(report)))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
