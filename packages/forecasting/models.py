"""Honest model ladder for reported-outbreak occurrence.

Three rungs, deliberately in this order:

1. :class:`SeasonalClimatologyBaseline` - the rate a district-week has carried a
   published outbreak report historically, smoothed over the calendar. Anything
   more complicated has to beat this to be published.
2. :class:`RidgeLogistic` - L2-penalised logistic regression fitted by IRLS on
   standardised environmental, calendar and reporting-history features, with the
   baseline log-odds supplied as a column so the model can only add information.
3. :class:`GradientBoostedStumps` - a histogram gradient-boosting challenger for
   non-linear structure.

Pure standard library on purpose: the platform ships to a small free host and
must not depend on a multi-hundred-megabyte numeric stack to reproduce its own
published evaluation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

CLIP = 1e-6
PROBABILITY_FLOOR = 5e-5
PROBABILITY_CEILING = 0.60


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


def logit(value: float) -> float:
    clipped = min(max(value, CLIP), 1.0 - CLIP)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class SeasonalClimatologyBaseline:
    """Multiplicative district x smoothed-week-of-year reporting climatology.

    ``p = global_rate * district_multiplier * week_multiplier``.  Both
    multipliers are shrunk towards 1, because 30 districts x 52 weeks of cells
    cannot be estimated from a few hundred published reports.
    """

    district_prior: float = 5.0
    week_prior: float = 1.0
    smoothing_half_width: int = 3
    global_rate: float = 0.0
    district_multiplier: dict[str, float] = None  # type: ignore[assignment]
    week_multiplier: dict[int, float] = None  # type: ignore[assignment]

    def fit(self, rows) -> SeasonalClimatologyBaseline:
        total = len(rows)
        if total == 0:
            raise ValueError("cannot fit a climatology on an empty training set")
        positives = sum(row.target for row in rows)
        self.global_rate = max((positives + 0.5) / (total + 1.0), PROBABILITY_FLOOR)

        district_events: dict[str, int] = defaultdict(int)
        district_weeks: dict[str, int] = defaultdict(int)
        week_events: dict[int, int] = defaultdict(int)
        week_weeks: dict[int, int] = defaultdict(int)
        for row in rows:
            district_events[row.district_id] += row.target
            district_weeks[row.district_id] += 1
            week_events[row.target_week_of_year] += row.target
            week_weeks[row.target_week_of_year] += 1

        self.district_multiplier = {}
        for district_id, exposure in district_weeks.items():
            expected = self.global_rate * exposure
            observed = district_events[district_id]
            self.district_multiplier[district_id] = (observed + self.district_prior) / (
                expected + self.district_prior
            )

        span = self.smoothing_half_width
        self.week_multiplier = {}
        weeks_present = sorted(week_weeks)
        highest = max(weeks_present) if weeks_present else 52
        for week in range(1, highest + 1):
            observed = 0
            exposure = 0
            for offset in range(-span, span + 1):
                neighbour = ((week - 1 + offset) % highest) + 1
                observed += week_events.get(neighbour, 0)
                exposure += week_weeks.get(neighbour, 0)
            expected = self.global_rate * exposure
            self.week_multiplier[week] = (observed + self.week_prior) / (
                expected + self.week_prior
            )
        return self

    def predict(self, rows) -> list[float]:
        output: list[float] = []
        for row in rows:
            district = self.district_multiplier.get(row.district_id, 1.0)
            week = self.week_multiplier.get(row.target_week_of_year, 1.0)
            probability = self.global_rate * district * week
            output.append(min(max(probability, PROBABILITY_FLOOR), PROBABILITY_CEILING))
        return output


def _standardise(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(matrix[0])
    count = len(matrix)
    means = [0.0] * width
    for row in matrix:
        for index in range(width):
            means[index] += row[index]
    means = [value / count for value in means]
    variances = [0.0] * width
    for row in matrix:
        for index in range(width):
            delta = row[index] - means[index]
            variances[index] += delta * delta
    deviations = [max(math.sqrt(value / count), 1e-9) for value in variances]
    return means, deviations


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[column][column] += 1e-9
            pivot = column
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row_index][index] -= factor * augmented[column][index]
    return [augmented[index][size] for index in range(size)]


@dataclass
class RidgeLogistic:
    """L2-penalised logistic regression fitted by iteratively reweighted least squares."""

    l2: float = 1.0
    max_iterations: int = 25
    tolerance: float = 1e-7
    coefficients: list[float] = None  # type: ignore[assignment]
    means: list[float] = None  # type: ignore[assignment]
    deviations: list[float] = None  # type: ignore[assignment]
    iterations: int = 0
    converged: bool = False

    def fit(self, matrix: list[list[float]], targets: list[int]) -> RidgeLogistic:
        if not matrix:
            raise ValueError("cannot fit a logistic model on an empty training set")
        self.means, self.deviations = _standardise(matrix)
        width = len(matrix[0])
        design = [
            [1.0] + [(row[i] - self.means[i]) / self.deviations[i] for i in range(width)]
            for row in matrix
        ]
        size = width + 1
        beta = [0.0] * size
        positives = sum(targets)
        beta[0] = logit(max((positives + 0.5) / (len(targets) + 1.0), CLIP))
        for iteration in range(self.max_iterations):
            hessian = [[0.0] * size for _ in range(size)]
            gradient = [0.0] * size
            for row, target in zip(design, targets, strict=True):
                linear = 0.0
                for index in range(size):
                    linear += beta[index] * row[index]
                probability = sigmoid(linear)
                weight = max(probability * (1.0 - probability), 1e-9)
                residual = target - probability
                for i in range(size):
                    value = row[i]
                    if value == 0.0:
                        continue
                    gradient[i] += residual * value
                    weighted = weight * value
                    row_i = hessian[i]
                    for j in range(i, size):
                        row_i[j] += weighted * row[j]
            for i in range(size):
                for j in range(i):
                    hessian[i][j] = hessian[j][i]
            for index in range(1, size):
                hessian[index][index] += self.l2
                gradient[index] -= self.l2 * beta[index]
            step = _solve(hessian, gradient)
            beta = [beta[index] + step[index] for index in range(size)]
            self.iterations = iteration + 1
            if max(abs(value) for value in step) < self.tolerance:
                self.converged = True
                break
        self.coefficients = beta
        return self

    def predict(self, matrix: list[list[float]]) -> list[float]:
        width = len(self.means)
        output: list[float] = []
        for row in matrix:
            linear = self.coefficients[0]
            for index in range(width):
                linear += self.coefficients[index + 1] * (
                    (row[index] - self.means[index]) / self.deviations[index]
                )
            output.append(
                min(max(sigmoid(linear), PROBABILITY_FLOOR), PROBABILITY_CEILING)
            )
        return output

    def standardised_coefficients(self, names: list[str]) -> dict[str, float]:
        return {
            name: round(self.coefficients[index + 1], 6)
            for index, name in enumerate(names)
        }


@dataclass(frozen=True, slots=True)
class _Node:
    feature: int = -1
    threshold: int = -1
    left: _Node | None = None
    right: _Node | None = None
    value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


@dataclass
class GradientBoostedTrees:
    """Histogram gradient boosting on log-loss; the non-linear challenger rung.

    Deliberately small (shallow trees, few rounds, quantile-binned features):
    the target has a base rate near one percent, so a large booster would only
    memorise which districts filed reports in the training years.
    """

    rounds: int = 60
    learning_rate: float = 0.10
    max_depth: int = 2
    bins: int = 12
    min_samples_leaf: int = 80
    l2: float = 5.0
    base_score: float = 0.0
    trees: list[_Node] = None  # type: ignore[assignment]
    thresholds: list[list[float]] = None  # type: ignore[assignment]

    def _bin_edges(self, matrix: list[list[float]]) -> list[list[float]]:
        width = len(matrix[0])
        edges: list[list[float]] = []
        for index in range(width):
            column = sorted(row[index] for row in matrix)
            cuts: list[float] = []
            for step in range(1, self.bins):
                position = min(max(int(len(column) * step / self.bins), 0), len(column) - 1)
                value = column[position]
                if not cuts or value > cuts[-1]:
                    cuts.append(value)
            edges.append(cuts)
        return edges

    def _binned(self, matrix: list[list[float]]) -> list[list[int]]:
        import bisect

        width = len(self.thresholds)
        return [
            [bisect.bisect_left(self.thresholds[i], row[i]) for i in range(width)]
            for row in matrix
        ]

    def _best_split(
        self,
        indices: list[int],
        binned: list[list[int]],
        gradients: list[float],
        hessians: list[float],
        width: int,
    ) -> tuple[int, int] | None:
        if len(indices) < 2 * self.min_samples_leaf:
            return None
        total_gradient = sum(gradients[i] for i in indices)
        total_hessian = sum(hessians[i] for i in indices)
        parent = total_gradient * total_gradient / (total_hessian + self.l2)
        best: tuple[int, int] | None = None
        best_gain = 1e-9
        span = self.bins + 1
        for feature in range(width):
            gradient_bins = [0.0] * span
            hessian_bins = [0.0] * span
            count_bins = [0] * span
            for i in indices:
                position = binned[i][feature]
                gradient_bins[position] += gradients[i]
                hessian_bins[position] += hessians[i]
                count_bins[position] += 1
            running_gradient = 0.0
            running_hessian = 0.0
            running_count = 0
            for threshold in range(span - 1):
                running_gradient += gradient_bins[threshold]
                running_hessian += hessian_bins[threshold]
                running_count += count_bins[threshold]
                right_count = len(indices) - running_count
                if running_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                right_gradient = total_gradient - running_gradient
                right_hessian = total_hessian - running_hessian
                gain = (
                    running_gradient * running_gradient / (running_hessian + self.l2)
                    + right_gradient * right_gradient / (right_hessian + self.l2)
                    - parent
                )
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold)
        return best

    def _leaf(self, indices: list[int], gradients: list[float], hessians: list[float]) -> _Node:
        gradient_sum = sum(gradients[i] for i in indices)
        hessian_sum = sum(hessians[i] for i in indices)
        return _Node(value=-gradient_sum / (hessian_sum + self.l2))

    def _grow(
        self,
        indices: list[int],
        binned: list[list[int]],
        gradients: list[float],
        hessians: list[float],
        width: int,
        depth: int,
    ) -> _Node:
        if depth >= self.max_depth or not indices:
            return self._leaf(indices, gradients, hessians) if indices else _Node()
        split = self._best_split(indices, binned, gradients, hessians, width)
        if split is None:
            return self._leaf(indices, gradients, hessians)
        feature, threshold = split
        left = [i for i in indices if binned[i][feature] <= threshold]
        right = [i for i in indices if binned[i][feature] > threshold]
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(left, binned, gradients, hessians, width, depth + 1),
            right=self._grow(right, binned, gradients, hessians, width, depth + 1),
        )

    @staticmethod
    def _apply(node: _Node, row: list[int]) -> float:
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
        return node.value

    def fit(self, matrix: list[list[float]], targets: list[int]) -> GradientBoostedTrees:
        if not matrix:
            raise ValueError("cannot fit a booster on an empty training set")
        self.thresholds = self._bin_edges(matrix)
        binned = self._binned(matrix)
        width = len(self.thresholds)
        positives = sum(targets)
        self.base_score = logit(max((positives + 0.5) / (len(targets) + 1.0), CLIP))
        scores = [self.base_score] * len(targets)
        self.trees = []
        everything = list(range(len(targets)))
        for _ in range(self.rounds):
            gradients: list[float] = []
            hessians: list[float] = []
            for score, target in zip(scores, targets, strict=True):
                probability = sigmoid(score)
                gradients.append(probability - target)
                hessians.append(max(probability * (1.0 - probability), 1e-9))
            tree = self._grow(everything, binned, gradients, hessians, width, 0)
            self.trees.append(tree)
            for index, row in enumerate(binned):
                scores[index] += self.learning_rate * self._apply(tree, row)
        return self

    def predict(self, matrix: list[list[float]]) -> list[float]:
        binned = self._binned(matrix)
        output: list[float] = []
        for row in binned:
            score = self.base_score
            for tree in self.trees:
                score += self.learning_rate * self._apply(tree, row)
            output.append(min(max(sigmoid(score), PROBABILITY_FLOOR), PROBABILITY_CEILING))
        return output
