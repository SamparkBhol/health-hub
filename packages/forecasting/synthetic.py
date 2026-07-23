"""Deterministic synthetic-only forecast harness.

This module tests issue-time feature construction, a named seasonal baseline,
rolling-origin evaluation, calibration diagnostics and map payloads.  It does
not consume Odisha observations and must never be described as model skill for
Odisha.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

SEED = 20260721
WATERMARK = "SIMULATION_ONLY_NOT_ODISHA_RISK"
DISTRICT_NAMES = (
    "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh",
    "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam",
    "Jagatsinghapur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal",
    "Kendrapara", "Keonjhar", "Khordha", "Koraput", "Malkangiri",
    "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri",
    "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh",
)


@dataclass(frozen=True)
class Example:
    district_id: str
    district_name: str
    issue_index: int
    issue_date: date
    target_date: date
    target_week_of_year: int
    features: tuple[float, ...]
    target: int


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _generate(
    seed: int = SEED, weeks: int = 156, horizon_weeks: int = 1
) -> list[Example]:
    if horizon_weeks not in {1, 2, 4, 8, 12}:
        raise ValueError("synthetic horizon must be one of 1, 2, 4, 8 or 12 weeks")
    rng = random.Random(seed)  # noqa: S311 - deterministic simulation, never security material
    start = date(2023, 1, 2)
    examples: list[Example] = []
    for district_index, district_name in enumerate(DISTRICT_NAMES, start=1):
        district_id = f"SIM-D{district_index:02d}"
        vulnerability = rng.uniform(-0.8, 0.8)
        rainfall: list[float] = []
        temperature: list[float] = []
        events: list[int] = []
        rainfall_outlooks: list[float] = []

        for week in range(weeks):
            phase = 2.0 * math.pi * (week % 52) / 52.0
            rain = max(0.0, 35.0 + 32.0 * math.sin(phase - 1.1) + rng.gauss(0, 8))
            temp = 27.0 + 4.0 * math.sin(phase - 0.25) + rng.gauss(0, 0.8)
            rainfall.append(rain)
            temperature.append(temp)
            rainfall_outlooks.append(max(0.0, rain + rng.gauss(0, 10)))
            lag = events[-1] if events else 0
            trailing = sum(events[-4:]) / max(1, len(events[-4:]))
            logit = (
                -3.25
                + vulnerability
                + 0.018 * rain
                + 0.07 * (temp - 27)
                + 0.8 * lag
                + 0.4 * trailing
            )
            events.append(int(rng.random() < _sigmoid(logit)))

        for issue in range(4, weeks - horizon_weeks):
            phase = 2.0 * math.pi * (issue % 52) / 52.0
            issue_date = start + timedelta(weeks=issue)
            target_date = issue_date + timedelta(weeks=horizon_weeks)
            features = (
                math.sin(phase),
                math.cos(phase),
                rainfall[issue],
                temperature[issue],
                float(events[issue]),
                sum(events[issue - 3 : issue + 1]) / 4.0,
                rainfall_outlooks[issue + horizon_weeks],
            )
            examples.append(
                Example(
                    district_id=district_id,
                    district_name=district_name,
                    issue_index=issue,
                    issue_date=issue_date,
                    target_date=target_date,
                    target_week_of_year=target_date.isocalendar().week,
                    features=features,
                    target=events[issue + horizon_weeks],
                )
            )
    return examples


def _standardise(
    train: list[Example], rows: list[Example]
) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(train[0].features)
    means = [sum(row.features[i] for row in train) / len(train) for i in range(width)]
    stds = []
    for index in range(width):
        variance = sum((row.features[index] - means[index]) ** 2 for row in train) / len(train)
        stds.append(max(math.sqrt(variance), 1e-8))
    matrix = [
        [(row.features[i] - means[i]) / stds[i] for i in range(width)] for row in rows
    ]
    return matrix, means, stds


def _fit_logistic(train: list[Example]) -> tuple[list[float], list[float], list[float]]:
    matrix, means, stds = _standardise(train, train)
    targets = [row.target for row in train]
    weights = [0.0] * (len(matrix[0]) + 1)
    rate = 0.08
    l2 = 0.02
    # Eighty deterministic full-batch steps are enough for this software-path
    # diagnostic and keep the free-host/CI runtime bounded.
    for _ in range(80):
        gradient = [0.0] * len(weights)
        for values, target in zip(matrix, targets, strict=True):
            linear = weights[0] + sum(
                weight * value
                for weight, value in zip(weights[1:], values, strict=True)
            )
            probability = _sigmoid(linear)
            error = probability - target
            gradient[0] += error
            for index, value in enumerate(values, start=1):
                gradient[index] += error * value
        scale = 1.0 / len(matrix)
        weights[0] -= rate * gradient[0] * scale
        for index in range(1, len(weights)):
            weights[index] -= rate * (gradient[index] * scale + l2 * weights[index])
    return weights, means, stds


def _predict_logistic(
    rows: list[Example], weights: list[float], means: list[float], stds: list[float]
) -> list[float]:
    return [
        _sigmoid(
            weights[0]
            + sum(
                weights[i + 1] * ((value - means[i]) / stds[i])
                for i, value in enumerate(row.features)
            )
        )
        for row in rows
    ]


def _seasonal_baseline(train: list[Example], rows: list[Example]) -> list[float]:
    by_week: dict[int, list[int]] = defaultdict(list)
    for row in train:
        by_week[row.target_week_of_year].append(row.target)
    overall = (sum(row.target for row in train) + 1.0) / (len(train) + 2.0)
    output = []
    for row in rows:
        values = by_week[row.target_week_of_year]
        output.append((sum(values) + 4.0 * overall) / (len(values) + 4.0))
    return output


def _brier(probabilities: Iterable[float], targets: Iterable[int]) -> float:
    pairs = list(zip(probabilities, targets, strict=True))
    return sum((probability - target) ** 2 for probability, target in pairs) / len(pairs)


def _reliability(probabilities: list[float], targets: list[int]) -> list[dict[str, float | int]]:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(5)]
    for probability, target in zip(probabilities, targets, strict=True):
        bins[min(4, int(probability * 5))].append((probability, target))
    report = []
    for index, values in enumerate(bins):
        report.append(
            {
                "lower": index / 5,
                "upper": (index + 1) / 5,
                "count": len(values),
                "mean_probability": (
                    round(sum(v[0] for v in values) / len(values), 6) if values else 0.0
                ),
                "observed_fraction": (
                    round(sum(v[1] for v in values) / len(values), 6) if values else 0.0
                ),
            }
        )
    return report


def build_synthetic_report(
    seed: int = SEED, horizon_weeks: int = 1
) -> dict[str, object]:
    examples = _generate(seed, horizon_weeks=horizon_weeks)
    origins = (104, 130, 143)
    all_model: list[float] = []
    all_baseline: list[float] = []
    all_targets: list[int] = []
    origin_reports = []
    for origin in origins:
        train = [row for row in examples if row.issue_index < origin]
        last_exclusive = 156 - horizon_weeks
        test = [
            row
            for row in examples
            if origin <= row.issue_index < min(origin + 8, last_exclusive)
        ]
        weights, means, stds = _fit_logistic(train)
        model = _predict_logistic(test, weights, means, stds)
        baseline = _seasonal_baseline(train, test)
        targets = [row.target for row in test]
        all_model.extend(model)
        all_baseline.extend(baseline)
        all_targets.extend(targets)
        origin_reports.append(
            {
                "origin_week_index": origin,
                "train_rows": len(train),
                "test_rows": len(test),
                "model_brier": round(_brier(model, targets), 6),
                "seasonal_baseline_brier": round(_brier(baseline, targets), 6),
            }
        )

    latest_index = max(row.issue_index for row in examples)
    train = [row for row in examples if row.issue_index < latest_index]
    latest = [row for row in examples if row.issue_index == latest_index]
    weights, means, stds = _fit_logistic(train)
    latest_probabilities = _predict_logistic(latest, weights, means, stds)
    map_values = [
        {
            "synthetic_district_id": row.district_id,
            "display_name": f"Synthetic {row.district_name}",
            "issue_date": row.issue_date.isoformat(),
            "target_date": row.target_date.isoformat(),
            "probability": round(probability, 6),
            "watermark": WATERMARK,
        }
        for row, probability in zip(latest, latest_probabilities, strict=True)
    ]
    return {
        "schema_version": "1.0.0",
        "watermark": WATERMARK,
        "is_synthetic": True,
        "seed": seed,
        "horizon_weeks": horizon_weeks,
        "target": f"synthetic_{horizon_weeks}_week_ahead_binary_event",
        "models": ["weekly_seasonal_naive", "l2_logistic_regression"],
        "rolling_origins": origin_reports,
        "pooled": {
            "model_brier": round(_brier(all_model, all_targets), 6),
            "seasonal_baseline_brier": round(_brier(all_baseline, all_targets), 6),
            "reliability": _reliability(all_model, all_targets),
            "note": "Software diagnostic only; not Odisha model performance or formal calibration.",
        },
        "latest_simulation_map": map_values,
        "real_odisha_prediction_available": False,
    }
