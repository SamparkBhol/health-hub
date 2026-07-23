"""Proper scoring rules, calibration diagnostics and block bootstrap.

Only strictly proper scores are used to decide publication (Brier and
logarithmic).  Discrimination summaries such as AUC are reported for context but
never gate publication, because a model can rank well and still be badly
calibrated - and a probability that is not calibrated is not usable for public
health decisions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

LOG_CLIP = 1e-6


def brier_score(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    pairs = list(zip(probabilities, targets, strict=True))
    if not pairs:
        raise ValueError("cannot score an empty evaluation set")
    return sum((probability - target) ** 2 for probability, target in pairs) / len(pairs)


def log_score(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    """Mean negative log-likelihood in nats (lower is better)."""

    pairs = list(zip(probabilities, targets, strict=True))
    if not pairs:
        raise ValueError("cannot score an empty evaluation set")
    total = 0.0
    for probability, target in pairs:
        clipped = min(max(probability, LOG_CLIP), 1.0 - LOG_CLIP)
        total -= math.log(clipped) if target else math.log(1.0 - clipped)
    return total / len(pairs)


def skill_score(model: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return 1.0 - model / reference


def auc(probabilities: Sequence[float], targets: Sequence[int]) -> float | None:
    positives = [p for p, y in zip(probabilities, targets, strict=True) if y == 1]
    negatives = [p for p, y in zip(probabilities, targets, strict=True) if y == 0]
    if not positives or not negatives:
        return None
    ordered = sorted(
        zip(probabilities, targets, strict=True), key=lambda item: item[0]
    )
    ranks: dict[int, float] = {}
    position = 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[position][0]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[index] = average
        position = end + 1
    positive_rank_sum = sum(
        ranks[index] for index, (_, target) in enumerate(ordered) if target == 1
    )
    count_positive = len(positives)
    count_negative = len(negatives)
    statistic = positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    return statistic / (count_positive * count_negative)


def reliability_bins(
    probabilities: Sequence[float], targets: Sequence[int], bins: int = 5
) -> list[dict[str, float | int]]:
    """Quantile-binned reliability curve.

    Fixed-width bins are useless when almost every forecast is below 0.05, so the
    curve is cut at forecast quantiles instead.
    """

    pairs = sorted(zip(probabilities, targets, strict=True), key=lambda item: item[0])
    total = len(pairs)
    if total == 0:
        return []
    size = max(1, total // bins)
    output: list[dict[str, float | int]] = []
    start = 0
    for index in range(bins):
        end = total if index == bins - 1 else min(total, start + size)
        chunk = pairs[start:end]
        start = end
        if not chunk:
            continue
        mean_probability = sum(item[0] for item in chunk) / len(chunk)
        observed = sum(item[1] for item in chunk) / len(chunk)
        output.append(
            {
                "bin": index,
                "count": len(chunk),
                "lower_forecast": round(chunk[0][0], 6),
                "upper_forecast": round(chunk[-1][0], 6),
                "mean_forecast": round(mean_probability, 6),
                "observed_frequency": round(observed, 6),
                "observed_events": sum(item[1] for item in chunk),
            }
        )
    return output


def expected_calibration_error(
    probabilities: Sequence[float], targets: Sequence[int], bins: int = 5
) -> float:
    curve = reliability_bins(probabilities, targets, bins)
    total = sum(int(item["count"]) for item in curve)
    if total == 0:
        return 0.0
    return round(
        sum(
            int(item["count"])
            * abs(float(item["mean_forecast"]) - float(item["observed_frequency"]))
            for item in curve
        )
        / total,
        6,
    )


def randomised_pit(
    probabilities: Sequence[float], targets: Sequence[int], *, seed: int = 20260721
) -> dict[str, object]:
    """Randomised probability integral transform for Bernoulli forecasts.

    For a calibrated forecaster the transform is uniform on [0, 1]; the histogram
    below is the binary analogue of a PIT plot.
    """

    generator = random.Random(seed)  # noqa: S311 - diagnostic randomisation, not security
    values: list[float] = []
    for probability, target in zip(probabilities, targets, strict=True):
        # F(y-1) and F(y) for a Bernoulli(p) predictive distribution.
        lower, upper = (1.0 - probability, 1.0) if target else (0.0, 1.0 - probability)
        values.append(lower + generator.random() * (upper - lower))
    counts = [0] * 10
    for value in values:
        counts[min(9, int(value * 10))] += 1
    total = len(values)
    expected = total / 10.0
    statistic = sum((count - expected) ** 2 / expected for count in counts) if total else 0.0
    return {
        "histogram": counts,
        "bin_count": 10,
        "sample_size": total,
        "uniformity_chi_square": round(statistic, 4),
        "degrees_of_freedom": 9,
        "note": (
            "Randomised PIT for binary outcomes; a calibrated forecaster gives a "
            "flat histogram. Rows are not independent across neighbouring weeks, "
            "so this is a diagnostic and not a formal test."
        ),
    }


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    blocks: int
    replicates: int
    mean_delta_brier: float
    lower_delta_brier: float
    upper_delta_brier: float
    mean_skill: float
    lower_skill: float
    upper_skill: float
    fraction_positive: float
    mean_delta_log: float
    lower_delta_log: float
    upper_delta_log: float
    fraction_log_positive: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "blocks": self.blocks,
            "replicates": self.replicates,
            "mean_delta_brier": round(self.mean_delta_brier, 8),
            "delta_brier_ci_2_5": round(self.lower_delta_brier, 8),
            "delta_brier_ci_97_5": round(self.upper_delta_brier, 8),
            "mean_brier_skill_score": round(self.mean_skill, 6),
            "brier_skill_ci_2_5": round(self.lower_skill, 6),
            "brier_skill_ci_97_5": round(self.upper_skill, 6),
            "fraction_of_replicates_model_better_brier": round(self.fraction_positive, 4),
            "mean_delta_log_score_nats": round(self.mean_delta_log, 8),
            "delta_log_score_ci_2_5": round(self.lower_delta_log, 8),
            "delta_log_score_ci_97_5": round(self.upper_delta_log, 8),
            "fraction_of_replicates_model_better_log_score": round(
                self.fraction_log_positive, 4
            ),
        }


def block_bootstrap(
    blocks: Sequence[tuple[list[float], list[float], list[int]]],
    *,
    replicates: int = 2000,
    seed: int = 20260721,
) -> BootstrapResult:
    """Resample whole blocks (seasons) with replacement.

    Neighbouring district-weeks are strongly dependent, so resampling individual
    rows would understate uncertainty by a large factor.  Blocks are whole
    evaluation seasons, which is the coarsest honest unit available.
    """

    if not blocks:
        raise ValueError("cannot bootstrap without blocks")
    generator = random.Random(seed)  # noqa: S311 - resampling, not security material
    # Pre-reduce each block once: summing per replicate would be quadratic waste.
    reduced: list[tuple[int, float, float, float, float]] = []
    for model, reference, targets in blocks:
        rows = len(targets)
        model_error = 0.0
        reference_error = 0.0
        model_log = 0.0
        reference_log = 0.0
        for probability, base, target in zip(model, reference, targets, strict=True):
            model_error += (probability - target) ** 2
            reference_error += (base - target) ** 2
            clipped = min(max(probability, LOG_CLIP), 1.0 - LOG_CLIP)
            clipped_base = min(max(base, LOG_CLIP), 1.0 - LOG_CLIP)
            if target:
                model_log -= math.log(clipped)
                reference_log -= math.log(clipped_base)
            else:
                model_log -= math.log(1.0 - clipped)
                reference_log -= math.log(1.0 - clipped_base)
        reduced.append((rows, model_error, reference_error, model_log, reference_log))

    deltas: list[float] = []
    skills: list[float] = []
    log_deltas: list[float] = []
    count = len(reduced)
    for _ in range(replicates):
        rows = 0
        model_error = 0.0
        reference_error = 0.0
        model_log = 0.0
        reference_log = 0.0
        for _index in range(count):
            block = reduced[generator.randrange(count)]
            rows += block[0]
            model_error += block[1]
            reference_error += block[2]
            model_log += block[3]
            reference_log += block[4]
        if rows == 0:
            continue
        model_brier = model_error / rows
        reference_brier = reference_error / rows
        deltas.append(reference_brier - model_brier)
        skills.append(skill_score(model_brier, reference_brier))
        log_deltas.append((reference_log - model_log) / rows)
    deltas.sort()
    skills.sort()
    log_deltas.sort()

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        position = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
        return values[position]

    return BootstrapResult(
        blocks=count,
        replicates=len(deltas),
        mean_delta_brier=sum(deltas) / len(deltas),
        lower_delta_brier=percentile(deltas, 0.025),
        upper_delta_brier=percentile(deltas, 0.975),
        mean_skill=sum(skills) / len(skills),
        lower_skill=percentile(skills, 0.025),
        upper_skill=percentile(skills, 0.975),
        fraction_positive=sum(1 for value in deltas if value > 0) / len(deltas),
        mean_delta_log=sum(log_deltas) / len(log_deltas),
        lower_delta_log=percentile(log_deltas, 0.025),
        upper_delta_log=percentile(log_deltas, 0.975),
        fraction_log_positive=sum(1 for value in log_deltas if value > 0) / len(log_deltas),
    )
