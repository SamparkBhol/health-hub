"""Read-only public disease layers and the three-month research outlook."""

from __future__ import annotations

import math
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from packages.forecasting.public_hmis import (
    FEATURE_NAMES,
    latest_district_history,
    load_model,
    ridge_probability,
    serving_features,
)
from pipelines.environmental.seasonal import load_seasonal_outlook
from pipelines.surveillance.hmis import load_hmis_rows
from pipelines.surveillance.ncvbdc import load_ncvbdc_rows


class PublicHealthDataError(RuntimeError):
    """A bundled public-health artefact is absent or invalid."""


@lru_cache(maxsize=1)
def _malaria_rows():
    rows = load_ncvbdc_rows()
    if not rows:
        raise PublicHealthDataError("bundled NCVBDC annual malaria data are unavailable")
    return rows


@lru_cache(maxsize=1)
def _hmis_rows():
    rows = load_hmis_rows()
    if not rows:
        raise PublicHealthDataError("bundled Odisha HMIS monthly data are unavailable")
    return rows


def malaria_annual_series() -> list[dict[str, int]]:
    """Statewide reported-malaria totals per year, oldest first.

    Summed across the 30 district rows of each year in the bundled official
    NCVBDC panel. These are reported cases/positives, so a change across years
    carries reporting and detection effort as well as transmission.
    """

    totals: dict[int, int] = {}
    for row in _malaria_rows():
        if row.total_cases is None:
            continue
        totals[row.year] = totals.get(row.year, 0) + int(row.total_cases)
    return [
        {"year": year, "total_cases": totals[year]} for year in sorted(totals)
    ]


def malaria_map(*, year: int | None = None, metric: str = "api") -> dict[str, Any]:
    if metric not in {"api", "total_cases", "spr", "aber", "pf_percent", "deaths"}:
        raise ValueError("unsupported NCVBDC malaria metric")
    all_rows = _malaria_rows()
    available_years = sorted({row.year for row in all_rows})
    selected_year = year or max(available_years)
    selected = [row for row in all_rows if row.year == selected_year]
    if not selected:
        raise ValueError(f"NCVBDC year {selected_year} is unavailable")
    records = []
    for row in selected:
        value = getattr(row, metric)
        records.append(
            {
                "district_id": row.district_id,
                "district_name": row.district_name,
                "year": row.year,
                "metric": metric,
                "value": value,
                "api": row.api,
                "total_cases": row.total_cases,
                "population_thousands": row.population_thousands,
                "aber": row.aber,
                "spr": row.spr,
                "pf_percent": row.pf_percent,
                "deaths": row.deaths,
                "observation_state": "observed" if value is not None else "not_reported_in_table",
                "source_url": row.source_url,
                "source_sha256": row.source_sha256,
            }
        )
    return {
        "status": "observed_public_data",
        "disease": "malaria",
        "year": selected_year,
        "available_years": available_years,
        "metric": metric,
        "metric_definition": {
            "api": "Annual Parasite Incidence as published by NCVBDC",
            "total_cases": "Annual total malaria cases/positives as published by NCVBDC",
            "spr": "Slide Positivity Rate as published by NCVBDC",
            "aber": "Annual Blood Examination Rate as published by NCVBDC",
            "pf_percent": "P. falciparum percentage as published by NCVBDC",
            "deaths": "Malaria deaths as published by NCVBDC",
        }[metric],
        "geography": "30 Odisha districts",
        "records": records,
        "source_scope": "official annual district malaria report; not a current weekly feed",
        # Stated explicitly rather than omitted: the badge contract keys on this
        # flag, and an absent field is ambiguous where False is a claim.
        "is_synthetic": False,
    }


def hmis_map(*, period: str | None = None, metric: str = "malaria_test_positivity"):
    allowed = {
        "malaria_microscopy_positive_records",
        "malaria_microscopy_positivity",
        "malaria_positive_records",
        "malaria_test_positivity",
        "dengue_positive_records",
        "childhood_diarrhoea_records",
    }
    if metric not in allowed:
        raise ValueError("unsupported HMIS metric")
    rows = _hmis_rows()
    periods = sorted({row.period_start for row in rows})
    selected_period = period or max(periods)
    selected = [row for row in rows if row.period_start == selected_period]
    if not selected:
        raise ValueError(f"HMIS period {selected_period} is unavailable")
    return {
        "status": "observed_public_data",
        "period": selected_period,
        "available_period_start": periods[0],
        "available_period_end": periods[-1],
        "metric": metric,
        "metric_scope": (
            "provisional facility-reported test/service records; not deduplicated "
            "people or population incidence"
        ),
        "records": [
            {
                "district_id": row.district_id,
                "district_name": row.district_name,
                "value": getattr(row, metric),
                "malaria_microscopy_positive_records": (
                    row.malaria_microscopy_positive_records
                ),
                "malaria_microscopy_tests": row.malaria_microscopy_tests,
                "malaria_microscopy_positivity": row.malaria_microscopy_positivity,
                "malaria_positive_records": row.malaria_positive_records,
                "malaria_tests": row.malaria_tests,
                "malaria_test_positivity": row.malaria_test_positivity,
                "dengue_positive_records": row.dengue_positive_records,
                "childhood_diarrhoea_records": row.childhood_diarrhoea_records,
                "observation_state": (
                    "observed" if getattr(row, metric) is not None else "not_reported"
                ),
                "source_url": row.source_url,
                "resource_url": row.resource_url,
                "source_sha256": row.source_sha256,
            }
            for row in selected
        ],
        "is_synthetic": False,
    }


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.values())
    denominator = max(len(ordered) - 1, 1)
    return {
        district_id: sum(item < value for item in ordered) / denominator
        for district_id, value in values.items()
    }


def _ridge_features(rain: float, temperature: float, month: int) -> list[float]:
    angle = 2.0 * math.pi * (month - 1) / 12.0
    return [math.log1p(max(rain, 0.0)), temperature, math.sin(angle), math.cos(angle)]


def _month_weights(start: date, end: date) -> dict[int, float]:
    """Day-weighted share of the target window falling in each calendar month.

    The seasonal windows are 30-day leads anchored on the day the ensemble file
    was refreshed, so they straddle two (occasionally three) calendar months.
    Reading a single month off the window midpoint made the served figure jump
    between calendar-month rates purely because of the refresh date; blending by
    the number of days actually in each month moves continuously instead.
    """

    if end < start:
        raise ValueError("target window ends before it starts")
    total_days = (end - start).days + 1
    counts: dict[int, int] = {}
    for offset in range(total_days):
        month = (start + timedelta(days=offset)).month
        counts[month] = counts.get(month, 0) + 1
    return {month: counts[month] / total_days for month in sorted(counts)}


def _statewide_month_rate(baseline: dict[str, Any], month: int) -> float:
    return float(baseline["month_rates"].get(str(month), baseline["global_rate"]))


def _blended_month_rate(baseline: dict[str, Any], weights: dict[int, float]) -> float:
    return sum(
        weight * _statewide_month_rate(baseline, month) for month, weight in weights.items()
    )


def _blended_district_context(
    baseline: dict[str, Any], district_id: str, weights: dict[int, float]
) -> float:
    cells = baseline["district_context_rates"]
    total = 0.0
    for month, weight in weights.items():
        cell = cells.get(f"{district_id}|{month}")
        value = float(cell) if cell is not None else _statewide_month_rate(baseline, month)
        total += weight * value
    return total


def _blended_ridge_probability(
    model: dict[str, Any],
    weights: dict[int, float],
    rain: float,
    temperature: float,
    history: dict[str, Any] | None,
) -> float | None:
    """Day-weighted mixture of the ridge probability over the months in the window.

    Returns None when the district has no observed lag history, because the
    fitted model reads that district's own recent excess and substituting a
    neutral value would invent an observation.
    """

    total = 0.0
    for month, weight in weights.items():
        features = serving_features(rain, temperature, month, history)
        if features is None:
            return None
        total += weight * ridge_probability(model, features)
    return total


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def _spearman(first: list[float], second: list[float]) -> float | None:
    """Rank correlation with average ranks for ties, or None if undefined."""

    if len(first) < 3 or len(first) != len(second):
        return None
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    count = len(first_ranks)
    mean_first = sum(first_ranks) / count
    mean_second = sum(second_ranks) / count
    covariance = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(first_ranks, second_ranks, strict=True)
    )
    variance_first = sum((value - mean_first) ** 2 for value in first_ranks)
    variance_second = sum((value - mean_second) ** 2 for value in second_ranks)
    if variance_first <= 0.0 or variance_second <= 0.0:
        return None
    return covariance / math.sqrt(variance_first * variance_second)


#: What the served central probability was actually produced by, recorded at the
#: point the value is assigned. Every published model/environment flag is derived
#: from this rather than from the artefact's selection label, so a different
#: winner cannot leave the labels describing something other than what is served.
ENVIRONMENT_SERVED_SOURCE = "ridge_logistic_environment_season_and_disease_history"
SERVED_SOURCE_MODEL = {
    ENVIRONMENT_SERVED_SOURCE: "ridge_logistic",
    "calendar_month_climatology_day_weighted": "calendar_month_baseline",
    "unconditional_climatology_constant": "unconditional_climatology",
}


def public_outlook_map(*, horizon_month: int = 1) -> dict[str, Any]:
    if horizon_month not in {1, 2, 3}:
        raise ValueError("horizon_month must be 1, 2 or 3")
    seasonal = load_seasonal_outlook()
    model = load_model()
    if seasonal is None or model is None:
        raise PublicHealthDataError("seasonal or public HMIS model artefact is unavailable")
    latest_year = max(row.year for row in _malaria_rows())
    latest = [row for row in _malaria_rows() if row.year == latest_year]
    burden = {row.district_id: row.api for row in latest}
    burden_ranks = _percentile_ranks(burden)
    annual_by_id = {row.district_id: row for row in latest}
    # The fitted model reads each district's own recent excess, so serving needs
    # that block per district. A district absent here abstains rather than being
    # scored from an invented neutral history.
    district_history = latest_district_history()
    baseline = model["serving_baseline"]
    artefact_selection = str(model["selected_by_brier"])
    use_environment = artefact_selection == "ridge_logistic"
    environment_withheld_reason: str | None = None
    if artefact_selection == "gradient_boosted_trees":
        # The compact artefact deliberately does not serialise the challenger.
        # A selected booster therefore fails safe to the evaluated baseline.
        environment_withheld_reason = "selected_challenger_is_not_serialised_in_the_artefact"
    elif not use_environment:
        environment_withheld_reason = "environment_block_was_not_selected_out_of_time"
    records: list[dict[str, Any]] = []
    for district in seasonal["districts"]:
        window = next(
            item
            for item in district["windows"]
            if int(item["horizon_month"]) == horizon_month
        )
        start = date.fromisoformat(str(window["start_date"]))
        end = date.fromisoformat(str(window["end_date"]))
        # Day-weighted over every calendar month the window touches. Picking one
        # month off the midpoint made the served number depend on the day the
        # seasonal file happened to be refreshed.
        weights = _month_weights(start, end)
        district_id = str(district["district_id"])
        district_context = _blended_district_context(baseline, district_id, weights)
        history = district_history.get(district_id)
        fitted_central = (
            _blended_ridge_probability(
                model,
                weights,
                float(window["precipitation_mean_mm"]),
                float(window["temperature_mean_c"]),
                history,
            )
            if use_environment
            else None
        )
        if use_environment and fitted_central is not None:
            central = fitted_central
            candidates = [
                value
                for rain in (
                    float(window["precipitation_p10_mm"]),
                    float(window["precipitation_p90_mm"]),
                )
                for temperature in (
                    float(window["temperature_p10_c"]),
                    float(window["temperature_p90_c"]),
                )
                if (
                    value := _blended_ridge_probability(
                        model, weights, rain, temperature, history
                    )
                )
                is not None
            ]
            lower: float | None = min(candidates) if candidates else None
            upper: float | None = max(candidates) if candidates else None
            interval_state = "propagated_from_ensemble_p10_p90_environment_scenarios"
            served_source = ENVIRONMENT_SERVED_SOURCE
        elif artefact_selection == "unconditional_climatology":
            # No model in the ladder beat a constant equal to the training base
            # rate, so the served figure is that constant. Serving a calendar-month
            # rate here would publish a curve the evaluation did not select.
            central = float(baseline["global_rate"])
            lower = None
            upper = None
            interval_state = "not_propagated_environment_is_not_in_the_probability"
            served_source = "unconditional_climatology_constant"
        else:
            central = _blended_month_rate(baseline, weights)
            lower = None
            upper = None
            interval_state = "not_propagated_environment_is_not_in_the_probability"
            served_source = "calendar_month_climatology_day_weighted"
        annual = annual_by_id[district_id]
        # A planning rank, never a probability: 80% latest official burden and
        # 20% the district's historical public-HMIS seasonal context.
        priority = 100.0 * (0.8 * burden_ranks[district_id] + 0.2 * district_context)
        records.append(
            {
                "district_id": district["district_id"],
                "district_name": district["district_name"],
                "horizon_month": horizon_month,
                "target_start": window["start_date"],
                "target_end": window["end_date"],
                "target_month_weights": {
                    str(month): round(weight, 6) for month, weight in weights.items()
                },
                "target_month_weighting": "share_of_window_days_per_calendar_month",
                "research_indicator_probability": round(central, 6),
                "served_probability_source": served_source,
                "probability_lower_environment_scenario": (
                    None if lower is None else round(lower, 6)
                ),
                "probability_upper_environment_scenario": (
                    None if upper is None else round(upper, 6)
                ),
                "probability_interval_state": interval_state,
                "surveillance_priority_score": round(priority, 2),
                "official_malaria_api": annual.api,
                "official_malaria_cases": annual.total_cases,
                "official_burden_year": latest_year,
                "historical_district_month_context": round(district_context, 6),
                "forecast_precipitation_mean_mm": window["precipitation_mean_mm"],
                "forecast_precipitation_p10_mm": window["precipitation_p10_mm"],
                "forecast_precipitation_p90_mm": window["precipitation_p90_mm"],
                "forecast_temperature_mean_c": window["temperature_mean_c"],
                "forecast_temperature_p10_c": window["temperature_p10_c"],
                "forecast_temperature_p90_c": window["temperature_p90_c"],
                "environment_used_in_probability": served_source == ENVIRONMENT_SERVED_SOURCE,
                "source_url": district["source_url"],
            }
        )
    records.sort(key=lambda item: float(item["surveillance_priority_score"]), reverse=True)

    # Derived from what was served, not from the artefact's selection label.
    served_sources = sorted({str(item["served_probability_source"]) for item in records})
    if len(served_sources) != 1:
        raise PublicHealthDataError("the served probability mixes models across districts")
    served_source = served_sources[0]
    # Whether the rainfall/temperature columns are inside the vector that produced
    # the served probability. Separate from whether they *earned* their place --
    # that is the ablation's verdict, read straight from the artefact below.
    environment_in_served_vector = served_source == ENVIRONMENT_SERVED_SOURCE
    ablation = model.get("environment_block_ablation") or {}
    environment_earns_its_place = bool(ablation.get("environment_reduces_brier"))
    environment_promoted = environment_in_served_vector and environment_earns_its_place
    selected_model = SERVED_SOURCE_MODEL[served_source]
    beats_unconditional = bool(model["beats_unconditional_climatology"])
    skill_score = model["brier_skill_score_vs_unconditional"]
    training_series_end = max(row.period_start for row in _hmis_rows())
    last_evaluated_year = int(str(model["origins"][-1])[:4])
    window_start = min(str(item["target_start"]) for item in records)
    window_end = max(str(item["target_end"]) for item in records)
    priority_values = [
        float(item["surveillance_priority_score"])
        for item in records
        if item["official_malaria_api"] is not None
    ]
    burden_values = [
        float(item["official_malaria_api"])
        for item in records
        if item["official_malaria_api"] is not None
    ]
    burden_correlation = _spearman(priority_values, burden_values)
    return {
        "status": "research_outlook",
        "disease": "malaria",
        "horizon_month": horizon_month,
        "forecast_target": model["target"],
        "not_target": model["not_target"],
        "selected_model": selected_model,
        "served_probability_source": served_source,
        "artefact_selected_by_brier": artefact_selection,
        "served_model_matches_artefact_selection": selected_model == artefact_selection,
        "environment_promoted": environment_promoted,
        "environment_in_served_vector": environment_in_served_vector,
        "environment_withheld_reason": environment_withheld_reason,
        "environment_ablation_result": (
            "reduced_brier_in_the_matched_ablation_and_is_retained"
            if environment_earns_its_place
            else "did_not_reduce_brier_against_the_same_model_without_it_"
            "and_is_carried_for_context_only"
        ),
        # The brief names environmental factors and historical disease information.
        # Both are in the served vector; only the second is measurably carrying the
        # skill, and saying so is the point of publishing the ablation.
        "skill_attribution": (
            "Lagged district disease history supplies the out-of-time skill. "
            "Removing the rainfall and temperature block from the same model "
            f"changes the pooled Brier score by {ablation.get('delta_brier')}, "
            "the same sign at every rolling origin."
        ),
        "environment_block_ablation": model.get("environment_block_ablation"),
        "beats_unconditional_climatology": beats_unconditional,
        "brier_skill_score_vs_unconditional": skill_score,
        "model_skill_statement": (
            "The selected model beat a constant equal to the training base rate "
            f"(Brier skill score {skill_score} against unconditional climatology)."
            if beats_unconditional
            else (
                "No model in the evaluated ladder beat a constant equal to the training "
                f"base rate (Brier skill score {skill_score} against unconditional "
                "climatology), so the served figure is that constant and carries no "
                "demonstrated skill beyond it."
            )
        ),
        "model_feature_names": list(FEATURE_NAMES),
        "model_evaluation": model["pooled"],
        "forecast_calibration_state": model["forecast_calibration_state"],
        "forecast_error_note": model["forecast_error_note"],
        "training_data_vintage": {
            "hmis_training_series_end": training_series_end,
            "last_evaluated_target_year": last_evaluated_year,
            "served_target_window_start": window_start,
            "served_target_window_end": window_end,
            "reason_code": "TRAINING_SERIES_ENDS_BEFORE_TARGET_WINDOW",
            "detail": (
                f"The HMIS training panel ends {training_series_end} and the last "
                f"rolling-origin year evaluated is {last_evaluated_year}, while this "
                f"window runs {window_start} to {window_end}. Nothing in the training "
                "data observes the served window."
            ),
        },
        "environment_provider": seasonal["provider"],
        "environment_model": seasonal["underlying_models"],
        "environment_generated_at": seasonal["generated_at"],
        "priority_definition": (
            f"0-100 surveillance-planning rank: 80% percentile of official {latest_year} "
            "NCVBDC API plus 20% district-month historical HMIS context; not probability"
        ),
        "priority_independence": {
            "method": "spearman_rank_correlation_computed_at_request_time",
            "compared_with": f"official NCVBDC annual API for {latest_year}",
            "districts_compared": len(priority_values),
            "coefficient": (
                None if burden_correlation is None else round(burden_correlation, 6)
            ),
            "interpretation": (
                "A coefficient at or near 1.0 means the priority ranking is largely a "
                "monotone re-expression of the observed official burden map and carries "
                "little information independent of it; the remainder comes from the 20% "
                "district-month historical HMIS context term."
            ),
        },
        "records": records,
        "is_synthetic": False,
    }


def public_outlook_evaluation() -> dict[str, Any]:
    model = load_model()
    if model is None:
        raise PublicHealthDataError("public HMIS model artefact is unavailable")
    return model

