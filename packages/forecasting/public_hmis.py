"""Research model for elevated public HMIS malaria microscopy positivity.

This model deliberately does *not* target an outbreak or disease incidence. Its
binary target is whether a district-month's microscopy positivity exceeds the
75th percentile of that district's preceding 24+ observed months.  It measures
whether rainfall, temperature and calendar season carry out-of-time information
about that public indicator.  Historical target-month weather is observed
reanalysis, not an archived forecast vintage, so forecast-error calibration is
outside this evaluation and current outputs remain research outlooks.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pipelines.environmental.historical import load_receipt, read_manifest
from pipelines.surveillance.hmis import HMISRow, load_hmis_rows

from .metrics import auc, brier_score, expected_calibration_error, log_score, reliability_bins
from .models import GradientBoostedTrees, RidgeLogistic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "forecasting" / "public_hmis_malaria_model.json"
FEATURE_NAMES = (
    "log_rain_30d",
    "temperature_30d",
    "month_sin",
    "month_cos",
    # Lagged disease history. The brief asks for prediction from environmental
    # factors *and historical disease information*; the second half was used only
    # to define the target and never as a predictor, which is why every model in
    # the ladder lost to a constant. Malaria positivity is strongly
    # autocorrelated, so a district's own recent excess is the single most
    # informative signal available at forecast time.
    "excess_lag1",
    "excess_lag2",
    "excess_lag3",
    "elevated_lag1",
    "elevated_lag2",
    "excess_mean_3m",
)
#: Feature-block indices into :attr:`PublicHMISModelRow.features`. The ablation
#: below fits each block alone and together, so "environment did not help" and
#: "disease history did" are measured comparisons rather than claims.
SEASON_FEATURE_INDICES = (2, 3)
ENVIRONMENT_FEATURE_INDICES = (0, 1)
HISTORY_FEATURE_INDICES = (4, 5, 6, 7, 8, 9)
#: Months of prior observation a row needs before it can carry the lag block.
LAG_HISTORY_MONTHS = 3
#: One penalty everywhere the ridge is fitted, including both arms of the
#: ablation. If each arm tuned its own penalty, a difference between them could
#: be a difference in tuning rather than in information.
RIDGE_L2 = 10.0
MINIMUM_HISTORY = 24
ELEVATED_QUANTILE = 0.75
#: Trailing months used for the served rate, so a declining series is not
#: forecast from its own high-event history.
_RECENT_WINDOW_MONTHS = 360


@dataclass(frozen=True, slots=True)
class PublicHMISModelRow:
    district_id: str
    period_start: date
    rain_mm: float
    temperature_c: float
    positivity: float
    threshold: float
    target: int
    #: log1p(positivity) - log1p(threshold) for the three preceding months, most
    #: recent first. Positive means that month ran above its own trailing
    #: 75th-percentile bar. Every value is observed strictly before this row.
    excess_lags: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Whether each of the two preceding months was itself an elevated month.
    elevated_lags: tuple[float, float] = (0.0, 0.0)

    @property
    def features(self) -> list[float]:
        angle = 2.0 * math.pi * (self.period_start.month - 1) / 12.0
        return [
            math.log1p(max(self.rain_mm, 0.0)),
            self.temperature_c,
            math.sin(angle),
            math.cos(angle),
            *self.excess_lags,
            *self.elevated_lags,
            sum(self.excess_lags) / len(self.excess_lags),
        ]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_monthly_climate() -> dict[tuple[str, str], tuple[float, float]]:
    """Return (district, YYYY-MM-01) -> (rain total, mean temperature)."""

    output: dict[tuple[str, str], tuple[float, float]] = {}
    for district_id, vintage in read_manifest().items():
        receipt = load_receipt(vintage)
        buckets: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
            lambda: {"PRECTOTCORR": [], "T2M": []}
        )
        for value in receipt.values:
            if value.is_fill_value or value.value is None:
                continue
            if value.parameter in {"PRECTOTCORR", "T2M"}:
                buckets[(value.day.year, value.day.month)][value.parameter].append(
                    float(value.value)
                )
        for (year, month), series in buckets.items():
            rain = series["PRECTOTCORR"]
            temperature = series["T2M"]
            if len(rain) < 28 or len(temperature) < 28:
                continue
            key = (district_id, f"{year:04d}-{month:02d}-01")
            output[key] = (sum(rain), sum(temperature) / len(temperature))
    return output


def build_model_rows(
    hmis_rows: list[HMISRow] | None = None,
    climate: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> list[PublicHMISModelRow]:
    selected = hmis_rows if hmis_rows is not None else load_hmis_rows()
    monthly_climate = climate if climate is not None else load_monthly_climate()
    by_district: dict[str, list[HMISRow]] = defaultdict(list)
    for row in selected:
        if row.malaria_microscopy_positivity is not None:
            by_district[row.district_id].append(row)
    output: list[PublicHMISModelRow] = []
    for district_id, rows in sorted(by_district.items()):
        history: list[float] = []
        # Each entry is (excess, elevated) for one observed month of this
        # district, most recent last. Only months already appended -- i.e.
        # strictly earlier than the row being built -- can enter its features,
        # which is what keeps the lag block leakage-free.
        observed: list[tuple[float, float]] = []
        for row in sorted(rows, key=lambda item: item.period_start):
            if row.malaria_microscopy_positivity is None:
                continue
            positivity = float(row.malaria_microscopy_positivity)
            weather = monthly_climate.get((district_id, row.period_start))
            threshold = (
                _quantile(history, ELEVATED_QUANTILE)
                if len(history) >= MINIMUM_HISTORY
                else None
            )
            if (
                weather is not None
                and threshold is not None
                and len(observed) >= LAG_HISTORY_MONTHS
            ):
                recent = observed[-LAG_HISTORY_MONTHS:][::-1]
                output.append(
                    PublicHMISModelRow(
                        district_id=district_id,
                        period_start=date.fromisoformat(row.period_start),
                        rain_mm=weather[0],
                        temperature_c=weather[1],
                        positivity=positivity,
                        threshold=threshold,
                        target=int(positivity > threshold),
                        excess_lags=(recent[0][0], recent[1][0], recent[2][0]),
                        elevated_lags=(recent[0][1], recent[1][1]),
                    )
                )
            if threshold is not None:
                observed.append(
                    (
                        math.log1p(max(positivity, 0.0))
                        - math.log1p(max(threshold, 0.0)),
                        float(positivity > threshold),
                    )
                )
            history.append(positivity)
    return output


class MonthBaseline:
    """Beta-smoothed calendar-month climatology fitted on training rows only."""

    def __init__(self) -> None:
        self.global_rate = 0.0
        self.month_rate: dict[int, float] = {}

    def fit(self, rows: list[PublicHMISModelRow]) -> MonthBaseline:
        positives = sum(row.target for row in rows)
        self.global_rate = (positives + 1.0) / (len(rows) + 2.0)
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            grouped[row.period_start.month].append(row.target)
        self.month_rate = {
            month: (sum(targets) + 5.0 * self.global_rate) / (len(targets) + 5.0)
            for month, targets in grouped.items()
        }
        return self

    def predict(self, rows: list[PublicHMISModelRow]) -> list[float]:
        return [self.month_rate.get(row.period_start.month, self.global_rate) for row in rows]


class DistrictMonthBaseline(MonthBaseline):
    """A higher-variance district x month challenger, strongly shrunk by month."""

    def __init__(self) -> None:
        super().__init__()
        self.district_month_rate: dict[tuple[str, int], float] = {}

    def fit(self, rows: list[PublicHMISModelRow]) -> DistrictMonthBaseline:
        super().fit(rows)
        cells: dict[tuple[str, int], list[int]] = defaultdict(list)
        for row in rows:
            cells[(row.district_id, row.period_start.month)].append(row.target)
        self.district_month_rate = {
            cell: (sum(targets) + 5.0 * self.month_rate[cell[1]]) / (len(targets) + 5.0)
            for cell, targets in cells.items()
        }
        return self

    def predict(self, rows: list[PublicHMISModelRow]) -> list[float]:
        return [
            self.district_month_rate.get(
                (row.district_id, row.period_start.month),
                self.month_rate.get(row.period_start.month, self.global_rate),
            )
            for row in rows
        ]


def _block(features: list[float], indices: tuple[int, ...]) -> list[float]:
    return [features[index] for index in indices]


def _scores(probabilities: list[float], targets: list[int]) -> dict[str, Any]:
    area = auc(probabilities, targets)
    return {
        "brier": round(brier_score(probabilities, targets), 8),
        "log_score": round(log_score(probabilities, targets), 8),
        "auc": None if area is None else round(area, 6),
        "expected_calibration_error": expected_calibration_error(probabilities, targets),
        "reliability": reliability_bins(probabilities, targets, bins=5),
    }


def train_and_evaluate(rows: list[PublicHMISModelRow] | None = None) -> dict[str, Any]:
    model_rows = rows if rows is not None else build_model_rows()
    origins = [date(2017, 1, 1), date(2018, 1, 1), date(2019, 1, 1)]
    folds: list[dict[str, Any]] = []
    pooled_targets: list[int] = []
    pooled_baseline: list[float] = []
    pooled_district_baseline: list[float] = []
    pooled_ridge: list[float] = []
    pooled_season_only_ridge: list[float] = []
    pooled_no_environment_ridge: list[float] = []
    pooled_no_history_ridge: list[float] = []
    pooled_persistence: list[float] = []
    pooled_booster: list[float] = []
    pooled_unconditional: list[float] = []
    for origin in origins:
        train = [row for row in model_rows if row.period_start < origin]
        test = [
            row
            for row in model_rows
            if origin <= row.period_start < date(origin.year + 1, 1, 1)
        ]
        if len(train) < 300 or not test:
            continue
        matrix = [row.features for row in train]
        targets = [row.target for row in train]
        test_matrix = [row.features for row in test]
        test_targets = [row.target for row in test]
        baseline = MonthBaseline().fit(train).predict(test)
        district_baseline = DistrictMonthBaseline().fit(train).predict(test)
        ridge_model = RidgeLogistic(l2=RIDGE_L2).fit(matrix, targets)
        ridge = ridge_model.predict(test_matrix)
        # Matched ablation arm: the same estimator, the same penalty and the same
        # fold, with the rainfall/temperature columns removed and *everything else
        # kept*. A block ablation is only attributable if the two arms differ by
        # exactly the block under test -- dropping the disease history at the same
        # time would credit the environment for the history block's contribution.
        no_environment_indices = SEASON_FEATURE_INDICES + HISTORY_FEATURE_INDICES
        no_environment_model = RidgeLogistic(l2=RIDGE_L2).fit(
            [_block(row, no_environment_indices) for row in matrix], targets
        )
        no_environment = no_environment_model.predict(
            [_block(row, no_environment_indices) for row in test_matrix]
        )
        # Season alone, kept as the floor of the ladder rather than as the
        # ablation counterfactual.
        season_only_model = RidgeLogistic(l2=RIDGE_L2).fit(
            [_block(row, SEASON_FEATURE_INDICES) for row in matrix], targets
        )
        season_only = season_only_model.predict(
            [_block(row, SEASON_FEATURE_INDICES) for row in test_matrix]
        )
        # Third ablation arm: season + environment with the lagged-disease block
        # removed. This is the arm that showed every earlier model losing to a
        # constant, so keeping it makes the improvement attributable.
        no_history_indices = SEASON_FEATURE_INDICES + ENVIRONMENT_FEATURE_INDICES
        no_history_model = RidgeLogistic(l2=RIDGE_L2).fit(
            [_block(row, no_history_indices) for row in matrix], targets
        )
        no_history = no_history_model.predict(
            [_block(row, no_history_indices) for row in test_matrix]
        )
        # Persistence: last month's elevated flag. The obvious competitor for an
        # autocorrelated series, and the one a reviewer will reach for first.
        persistence = [
            0.85 if row.elevated_lags[0] > 0.5 else 0.05 for row in test
        ]
        booster_model = GradientBoostedTrees(
            rounds=40, max_depth=2, min_samples_leaf=60, l2=8.0
        ).fit(matrix, targets)
        booster = booster_model.predict(test_matrix)
        # The null competitor: predict the training base rate every month. It uses
        # strictly less information than any model in the ladder and is available at
        # every origin, so omitting it let a model be crowned without facing the one
        # baseline that can beat it.
        train_rate = sum(targets) / len(targets)
        pooled_unconditional.extend([train_rate] * len(test_targets))
        pooled_targets.extend(test_targets)
        pooled_baseline.extend(baseline)
        pooled_district_baseline.extend(district_baseline)
        pooled_ridge.extend(ridge)
        pooled_season_only_ridge.extend(season_only)
        pooled_no_environment_ridge.extend(no_environment)
        pooled_no_history_ridge.extend(no_history)
        pooled_persistence.extend(persistence)
        pooled_booster.extend(booster)
        folds.append(
            {
                "origin": origin.isoformat(),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_events": sum(test_targets),
                "baseline_brier": round(brier_score(baseline, test_targets), 8),
                "district_month_baseline_brier": round(
                    brier_score(district_baseline, test_targets), 8
                ),
                "ridge_brier": round(brier_score(ridge, test_targets), 8),
                "season_only_ridge_brier": round(brier_score(season_only, test_targets), 8),
                "no_environment_ridge_brier": round(
                    brier_score(no_environment, test_targets), 8
                ),
                "booster_brier": round(brier_score(booster, test_targets), 8),
            }
        )

    baseline_scores = _scores(pooled_baseline, pooled_targets)
    district_baseline_scores = _scores(pooled_district_baseline, pooled_targets)
    ridge_scores = _scores(pooled_ridge, pooled_targets)
    season_only_scores = _scores(pooled_season_only_ridge, pooled_targets)
    no_environment_scores = _scores(pooled_no_environment_ridge, pooled_targets)
    no_history_scores = _scores(pooled_no_history_ridge, pooled_targets)
    persistence_scores = _scores(pooled_persistence, pooled_targets)
    booster_scores = _scores(pooled_booster, pooled_targets)
    unconditional_scores = _scores(pooled_unconditional, pooled_targets)
    # The matched environment-block ablation. Both arms use the same rows, the same
    # rolling origins, the same estimator and the same penalty; only the feature
    # block differs, so the gap is attributable to the environment columns.
    with_environment_brier = float(ridge_scores["brier"])
    no_environment_brier = float(no_environment_scores["brier"])
    environment_block_ablation: dict[str, Any] = {
        "design": (
            "identical rolling origins, identical rows, identical RidgeLogistic "
            f"estimator at a single fixed l2={RIDGE_L2}; the two arms differ by the "
            "environment block alone, so the gap is attributable to it"
        ),
        "with_environment": {
            "features": list(FEATURE_NAMES),
            "brier": ridge_scores["brier"],
            "log_score": ridge_scores["log_score"],
            "auc": ridge_scores["auc"],
        },
        "without_environment": {
            "features": [FEATURE_NAMES[index] for index in no_environment_indices],
            "brier": no_environment_scores["brier"],
            "log_score": no_environment_scores["log_score"],
            "auc": no_environment_scores["auc"],
        },
        # Season alone is reported for context only. It is not the counterfactual:
        # it drops two blocks at once and would credit the environment with the
        # disease-history block's contribution.
        "season_only_reference": {
            "features": [FEATURE_NAMES[index] for index in SEASON_FEATURE_INDICES],
            "brier": season_only_scores["brier"],
            "log_score": season_only_scores["log_score"],
            "auc": season_only_scores["auc"],
        },
        "removed_features": [FEATURE_NAMES[index] for index in ENVIRONMENT_FEATURE_INDICES],
        "delta_brier": round(with_environment_brier - no_environment_brier, 8),
        "environment_reduces_brier": with_environment_brier < no_environment_brier,
        # Per-fold deltas, so the pooled verdict cannot rest on a single origin.
        "delta_brier_by_origin": {
            fold["origin"]: round(
                float(fold["ridge_brier"]) - float(fold["no_environment_ridge_brier"]), 8
            )
            for fold in folds
        },
        "interpretation": (
            "delta_brier is with_environment minus without_environment, both arms "
            "carrying calendar season and lagged disease history. Negative means the "
            "rainfall and temperature block lowered the pooled Brier score; positive "
            "or ~0 means it bought nothing on top of season and history."
        ),
    }
    selected = min(
        (
            ("calendar_month_baseline", baseline_scores),
            ("district_calendar_month_baseline", district_baseline_scores),
            ("ridge_logistic", ridge_scores),
            ("gradient_boosted_trees", booster_scores),
            ("unconditional_climatology", unconditional_scores),
            ("persistence_previous_month", persistence_scores),
            ("season_environment_no_history", no_history_scores),
        ),
        key=lambda item: float(item[1]["brier"]),
    )[0]
    # The expanding-percentile target is applied to a sharply declining series, so
    # the event rate is non-stationary by construction: it falls from ~0.57 in 2014
    # to ~0.07 by the last evaluated years. A rate fitted across the whole span
    # therefore over-forecasts the present regime, so serving uses a trailing window
    # and the per-year rates are published so a reader can see the drift.
    ordered_rows = sorted(model_rows, key=lambda row: row.period_start)
    recent_rows = ordered_rows[-_RECENT_WINDOW_MONTHS:] or ordered_rows
    recent_rate = (sum(row.target for row in recent_rows) + 1.0) / (
        len(recent_rows) + 2.0
    )
    rate_by_year: dict[int, list[int]] = defaultdict(list)
    for row in model_rows:
        rate_by_year[row.period_start.year].append(row.target)
    target_base_rate_by_year = {
        str(year): {
            "rows": len(targets),
            "events": sum(targets),
            "rate": round(sum(targets) / len(targets), 6),
        }
        for year, targets in sorted(rate_by_year.items())
    }
    pooled_base_rate = sum(pooled_targets) / len(pooled_targets)
    served_mean_forecast = sum(pooled_unconditional) / len(pooled_unconditional)

    final_ridge = RidgeLogistic(l2=RIDGE_L2).fit(
        [row.features for row in model_rows], [row.target for row in model_rows]
    )
    final_baseline = MonthBaseline().fit(model_rows)
    final_district_baseline = DistrictMonthBaseline().fit(model_rows)
    train_probabilities = final_ridge.predict([row.features for row in model_rows])
    train_brier = brier_score(train_probabilities, [row.target for row in model_rows])
    test_brier = float(ridge_scores["brier"])
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "target": (
            "district-month HMIS malaria microscopy positivity above the expanding "
            "district-specific 75th percentile after at least 24 prior observations"
        ),
        "not_target": "disease incidence, unique patients, or an operational outbreak",
        "modeling_rows": len(model_rows),
        "events": sum(row.target for row in model_rows),
        "feature_names": list(FEATURE_NAMES),
        "origins": [item.isoformat() for item in origins],
        "folds": folds,
        "pooled": {
            "unconditional_climatology": unconditional_scores,
            "persistence_previous_month": persistence_scores,
            "season_environment_no_history": no_history_scores,
            "calendar_month_baseline": baseline_scores,
            "district_calendar_month_baseline": district_baseline_scores,
            "ridge_logistic": ridge_scores,
            "gradient_boosted_trees": booster_scores,
        },
        "selected_by_brier": selected,
        "environment_block_ablation": environment_block_ablation,
        "ridge_beats_baseline": float(ridge_scores["brier"]) < float(baseline_scores["brier"]),
        # Skill against the null competitor, stated plainly. A negative score means
        # every model in the ladder is beaten by predicting the training base rate
        # every month, which a reader must be able to see without recomputing it.
        "beats_unconditional_climatology": float(
            min(
                baseline_scores["brier"],
                district_baseline_scores["brier"],
                ridge_scores["brier"],
                booster_scores["brier"],
            )
        )
        < float(unconditional_scores["brier"]),
        "target_base_rate_by_year": target_base_rate_by_year,
        "pooled_base_rate": round(pooled_base_rate, 6),
        "served_mean_forecast": round(served_mean_forecast, 6),
        "over_forecast_ratio": round(served_mean_forecast / pooled_base_rate, 3)
        if pooled_base_rate else None,
        "brier_skill_score_vs_unconditional": round(
            1.0
            - float(dict(
                calendar_month_baseline=baseline_scores,
                district_calendar_month_baseline=district_baseline_scores,
                ridge_logistic=ridge_scores,
                gradient_boosted_trees=booster_scores,
                unconditional_climatology=unconditional_scores,
            )[selected]["brier"])
            / float(unconditional_scores["brier"]),
            6,
        ),
        "overfit_diagnostic": {
            "ridge_train_brier": round(train_brier, 8),
            "ridge_rolling_origin_brier": round(test_brier, 8),
            "absolute_gap": round(abs(test_brier - train_brier), 8),
        },
        "serving_model": {
            "kind": "ridge_logistic",
            "l2": final_ridge.l2,
            "coefficients": final_ridge.coefficients,
            "means": final_ridge.means,
            "deviations": final_ridge.deviations,
            "converged": final_ridge.converged,
            "iterations": final_ridge.iterations,
        },
        "serving_baseline": {
            "kind": "calendar_month_climatology",
            # Fitted across every modelled month, which is dominated by the
            # high-event early years. Serving it would over-forecast the recent
            # regime roughly fourfold, so `recent_global_rate` below is what the
            # serving path uses; both are published so the gap is visible.
            "global_rate": recent_rate,
            "all_years_global_rate": final_baseline.global_rate,
            "recent_window_months": _RECENT_WINDOW_MONTHS,
            "month_rates": {
                str(month): probability
                for month, probability in sorted(final_baseline.month_rate.items())
            },
            "district_context_rates": {
                f"{district_id}|{month}": probability
                for (district_id, month), probability in sorted(
                    final_district_baseline.district_month_rate.items()
                )
            },
        },
        "forecast_calibration_state": "not_calibrated_for_current_outbreak_probability",
        "forecast_error_note": (
            "Backtests use observed target-month NASA POWER weather. Historical seasonal "
            "forecast vintages are unavailable, so EC46/SEAS5 forecast error is not part "
            "of the reported scores."
        ),
    }
    return payload


def write_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    payload = train_and_evaluate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_district_history() -> dict[str, dict[str, Any]]:
    """The most recent observed lag block per district, for serving.

    The fitted model reads a district's own recent excess, so serving needs the
    same block. This returns the last modelled row per district -- the newest
    month for which the panel actually has an observation -- together with the
    date it came from, so a caller can state how stale the history is rather
    than implying it is current.
    """

    latest: dict[str, PublicHMISModelRow] = {}
    for row in build_model_rows():
        held = latest.get(row.district_id)
        if held is None or row.period_start > held.period_start:
            latest[row.district_id] = row
    output: dict[str, dict[str, Any]] = {}
    for district_id, row in latest.items():
        # Roll the window forward by one month: the row's own outcome becomes the
        # most recent lag, exactly as build_model_rows does for the next month.
        excess = math.log1p(max(row.positivity, 0.0)) - math.log1p(
            max(row.threshold, 0.0)
        )
        output[district_id] = {
            "observed_through": row.period_start.isoformat(),
            "excess_lags": (excess, row.excess_lags[0], row.excess_lags[1]),
            "elevated_lags": (float(row.target), row.elevated_lags[0]),
        }
    return output


def serving_features(
    rain: float, temperature: float, month: int, history: dict[str, Any] | None
) -> list[float] | None:
    """Full feature vector for one district-month, or None without history.

    Returning None rather than zero-filling is deliberate: a zero lag block is a
    confident claim that the district ran exactly at its own threshold, which is
    a fabricated observation. A district with no history must abstain instead.
    """

    if history is None:
        return None
    angle = 2.0 * math.pi * (month - 1) / 12.0
    excess = tuple(float(value) for value in history["excess_lags"])
    elevated = tuple(float(value) for value in history["elevated_lags"])
    return [
        math.log1p(max(rain, 0.0)),
        temperature,
        math.sin(angle),
        math.cos(angle),
        *excess,
        *elevated,
        sum(excess) / len(excess),
    ]


def ridge_probability(model_payload: dict[str, Any], features: list[float]) -> float:
    fitted = model_payload["serving_model"]
    coefficients = [float(value) for value in fitted["coefficients"]]
    means = [float(value) for value in fitted["means"]]
    deviations = [float(value) for value in fitted["deviations"]]
    linear = coefficients[0]
    for index, value in enumerate(features):
        linear += coefficients[index + 1] * ((value - means[index]) / deviations[index])
    if linear >= 0:
        return 1.0 / (1.0 + math.exp(-min(linear, 60.0)))
    exponent = math.exp(max(linear, -60.0))
    return exponent / (1.0 + exponent)
