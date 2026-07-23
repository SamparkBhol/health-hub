"""Experimental environmental feature index for current district conditions.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The historical EpiClim catalogue-row experiment
(:mod:`packages.forecasting.backtest`) can only speak about weeks the EpiClim
catalogue covers, so it can say nothing at all about *now*.  This module fills
exactly that gap, and only that gap.

It fits one thing on the real historical panel: the weak association between the
environmental block and EpiClim row occurrence, with the district x week-of-year
catalogue climatology included as an ordinary covariate. The fitted environmental
coefficients define a single scalar,

    E[district, week] = sum_i beta_i * z_i(district, week)

over the environmental columns only - no calendar term, no reporting history, no
climatology intercept.  ``E`` is the part of the historical fit that came from
weather and nothing else.

The published number is then a **percentile**: where this week's ``E`` falls in
the distribution of ``E`` that this same district experienced in the same part
of the calendar across 2009-2022.  A value of 92 means "the environmental
index for this district right now is above 92 percent of the mid-Julys this
district has had since 2009". It does not establish biological favourability.

It is therefore:

* NOT a probability that an outbreak will occur;
* NOT a probability that a report will be published;
* NOT incidence and NOT a case count;
* NOT a forecast of anything - it describes conditions that have already
  happened, measured up to the last day the climate provider has published.

The environment ablation is approximately null. This index is retained only as
experimental risk-factor context; relabelling it as outbreak risk or
transmission suitability would be unsupported.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = REPO_ROOT / "data" / "forecasting" / "environmental_suitability_model.json"
SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "experimental-environmental-feature-index-1.0.0"

QUANTITY = "experimental_environmental_feature_index_percentile"
QUANTITY_STATEMENT = (
    "Experimental percentile of a district's current environmental feature index "
    "within its own 2009-2022 history for the same part of the calendar. The weak "
    "weights come from an association with EpiClim file-row occurrence and have "
    "not demonstrated disease-prediction value. It describes weather context, not "
    "transmission suitability, outbreak risk, incidence or case counts."
)

#: Half-width, in ISO weeks, of the seasonal window a percentile is taken over.
SEASONAL_HALF_WIDTH = 3
#: Quantile grid persisted per district x ISO week.
QUANTILE_GRID: tuple[float, ...] = tuple(round(index / 20.0, 2) for index in range(21))
#: A reference distribution thinner than this cannot support a percentile.
MINIMUM_REFERENCE_SAMPLES = 30

BANDS: tuple[tuple[float, str], ...] = (
    (50.0, "below_typical"),
    (75.0, "typical"),
    (90.0, "elevated"),
    (100.1, "much_above_typical"),
)


class SuitabilityArtefactMissing(RuntimeError):
    """No fitted suitability artefact is available."""


class SuitabilityArtefactInvalid(RuntimeError):
    """The suitability artefact exists but failed its invariant checks."""


def band_for(percentile: float) -> str:
    for ceiling, label in BANDS:
        if percentile < ceiling:
            return label
    return BANDS[-1][1]


def _quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    last = len(ordered) - 1
    output: list[float] = []
    for level in QUANTILE_GRID:
        position = level * last
        low = int(position)
        high = min(low + 1, last)
        weight = position - low
        output.append(round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 6))
    return output


def _percentile_from_quantiles(quantiles: list[float], value: float) -> float:
    """Invert a monotone quantile grid; clamps at the observed extremes."""

    if not quantiles:
        raise ValueError("empty quantile grid")
    if value <= quantiles[0]:
        return 0.0
    if value >= quantiles[-1]:
        return 100.0
    index = bisect_left(quantiles, value)
    low = quantiles[index - 1]
    high = quantiles[index]
    span = high - low
    fraction = 0.0 if span <= 0 else (value - low) / span
    level = QUANTILE_GRID[index - 1] + fraction * (QUANTILE_GRID[index] - QUANTILE_GRID[index - 1])
    return round(level * 100.0, 2)


def _seasonal_key(iso_week: int) -> int:
    return max(1, min(53, iso_week))


@dataclass(frozen=True, slots=True)
class SuitabilityScore:
    district_id: str
    iso_week: int
    linear_predictor: float
    percentile: float | None
    band: str | None
    reference_samples: int
    drivers: tuple[dict[str, Any], ...]
    status: str
    reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "district_id": self.district_id,
            "iso_week": self.iso_week,
            "environmental_linear_predictor": round(self.linear_predictor, 6),
            "suitability_percentile": self.percentile,
            "band": self.band,
            "reference_samples": self.reference_samples,
            "top_drivers": list(self.drivers),
            "status": self.status,
            "reason_code": self.reason_code,
        }


class SuitabilityModel:
    """Read-side scorer for the fitted environmental suitability artefact."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SuitabilityArtefactInvalid(
                f"unexpected schema_version {payload.get('schema_version')!r}"
            )
        if payload.get("is_synthetic") is not False:
            raise SuitabilityArtefactInvalid("suitability artefact must declare is_synthetic=false")
        quantity = payload.get("quantity", {})
        if quantity.get("is_incidence") is not False or quantity.get("is_case_count") is not False:
            raise SuitabilityArtefactInvalid(
                "artefact must declare that it is neither incidence nor a case count"
            )
        self.payload = payload
        self.feature_names: list[str] = list(payload["features"]["names"])
        self.coefficients: list[float] = [float(v) for v in payload["fit"]["coefficients"]]
        self.means: list[float] = [float(v) for v in payload["fit"]["means"]]
        self.deviations: list[float] = [float(v) for v in payload["fit"]["deviations"]]
        if not (
            len(self.feature_names)
            == len(self.coefficients)
            == len(self.means)
            == len(self.deviations)
        ):
            raise SuitabilityArtefactInvalid("fit vectors disagree on width")
        self.reference: dict[str, dict[str, list[float]]] = payload["reference_distribution"]

    @property
    def generated_at(self) -> str:
        return str(self.payload.get("generated_at", ""))

    def linear_predictor(
        self, features: list[float] | tuple[float, ...]
    ) -> tuple[float, list[dict[str, Any]]]:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} environmental features, got {len(features)}"
            )
        total = 0.0
        contributions: list[dict[str, Any]] = []
        for index, name in enumerate(self.feature_names):
            z = (float(features[index]) - self.means[index]) / self.deviations[index]
            term = self.coefficients[index] * z
            total += term
            contributions.append(
                {
                    "feature": name,
                    "value": round(float(features[index]), 4),
                    "standardised": round(z, 4),
                    "coefficient": round(self.coefficients[index], 6),
                    "contribution": round(term, 6),
                }
            )
        contributions.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        return total, contributions

    def score(
        self,
        district_id: str,
        iso_week: int,
        features: list[float] | tuple[float, ...],
        *,
        driver_count: int = 4,
    ) -> SuitabilityScore:
        total, contributions = self.linear_predictor(features)
        district_reference = self.reference.get(district_id, {})
        key = str(_seasonal_key(iso_week))
        entry = district_reference.get(key)
        if not entry:
            return SuitabilityScore(
                district_id=district_id,
                iso_week=iso_week,
                linear_predictor=total,
                percentile=None,
                band=None,
                reference_samples=0,
                drivers=tuple(contributions[:driver_count]),
                status="insufficient_reference",
                reason_code="NO_SEASONAL_REFERENCE_DISTRIBUTION_FOR_DISTRICT_WEEK",
            )
        samples = int(entry[0])
        quantiles = [float(value) for value in entry[1:]]
        if samples < MINIMUM_REFERENCE_SAMPLES:
            return SuitabilityScore(
                district_id=district_id,
                iso_week=iso_week,
                linear_predictor=total,
                percentile=None,
                band=None,
                reference_samples=samples,
                drivers=tuple(contributions[:driver_count]),
                status="insufficient_reference",
                reason_code="REFERENCE_DISTRIBUTION_BELOW_MINIMUM_SAMPLES",
            )
        percentile = _percentile_from_quantiles(quantiles, total)
        return SuitabilityScore(
            district_id=district_id,
            iso_week=iso_week,
            linear_predictor=total,
            percentile=percentile,
            band=band_for(percentile),
            reference_samples=samples,
            drivers=tuple(contributions[:driver_count]),
            status="scored",
        )


def load_model(path: Path | None = None) -> SuitabilityModel:
    target = path or ARTEFACT_PATH
    if not target.exists():
        raise SuitabilityArtefactMissing(
            f"no fitted environmental suitability artefact at {target}; run "
            "`python -m packages.forecasting.suitability`"
        )
    return SuitabilityModel(json.loads(target.read_text(encoding="utf-8")))


def fit_suitability_model(
    *,
    group: str = "any_reported_outbreak",
    l2: float = 8.0,
    progress=None,
) -> dict[str, Any]:
    """Fit the environmental block against EpiClim catalogue-row occurrence.

    The seasonal catalogue climatology enters as an estimated covariate. It is
    not a fixed statistical offset; the fitted coefficient is retained in the
    artefact so that distinction is auditable.
    """

    from .backtest import PANEL_END, PANEL_START, variant_columns
    from .climate import FEATURE_NAMES as ENVIRONMENT_FEATURE_NAMES
    from .climate import FEATURE_NOTES, build_feature_index, load_weekly_panel
    from .models import RidgeLogistic, SeasonalClimatologyBaseline, logit
    from .panel import build_examples, panel_weeks
    from .target import TARGET_KIND, TARGET_STATEMENT, build_target_panel

    weekly = load_weekly_panel()
    climate = build_feature_index(weekly)
    weeks = panel_weeks(PANEL_START, PANEL_END)
    target_panel = build_target_panel(group)
    rows = build_examples(target_panel=target_panel, climate=climate, horizon_weeks=1, weeks=weeks)
    if not rows:
        raise RuntimeError("no modelling rows; the climate cache or catalogue is missing")
    if progress:
        progress(f"built {len(rows)} rows")

    environment_columns = list(range(len(ENVIRONMENT_FEATURE_NAMES)))
    baseline = SeasonalClimatologyBaseline().fit(rows)
    offsets = baseline.predict(rows)
    design = [
        [*(row.features[index] for index in environment_columns), logit(probability)]
        for row, probability in zip(rows, offsets, strict=True)
    ]
    targets = [row.target for row in rows]
    model = RidgeLogistic(l2=l2).fit(design, targets)
    if progress:
        progress(f"fitted ridge logistic (converged={model.converged})")

    width = len(environment_columns)
    means = model.means[:width]
    deviations = model.deviations[:width]
    coefficients = model.coefficients[1 : width + 1]

    # Historical distribution of the environment-only linear predictor.
    per_district_week: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        total = 0.0
        for index in range(width):
            z = (row.features[index] - means[index]) / deviations[index]
            total += coefficients[index] * z
        iso_week = row.issue_week.isocalendar().week
        bucket = per_district_week.setdefault(row.district_id, {})
        for offset in range(-SEASONAL_HALF_WIDTH, SEASONAL_HALF_WIDTH + 1):
            neighbour = ((iso_week - 1 + offset) % 52) + 1
            bucket.setdefault(neighbour, []).append(total)

    reference: dict[str, dict[str, list[float]]] = {}
    for district_id, buckets in sorted(per_district_week.items()):
        district_reference: dict[str, list[float]] = {}
        for iso_week, values in sorted(buckets.items()):
            district_reference[str(iso_week)] = [float(len(values)), *_quantiles(values)]
        # ISO week 53 reuses week 52's reference; the calendar supplies it in
        # only a handful of years, never enough for its own distribution.
        if "52" in district_reference:
            district_reference["53"] = district_reference["52"]
        reference[district_id] = district_reference

    variant_width = len(variant_columns("environment_and_reporting_history"))
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "is_synthetic": False,
        "uses_real_odisha_data": True,
        "quantity": {
            "kind": QUANTITY,
            "statement": QUANTITY_STATEMENT,
            "is_incidence": False,
            "is_case_count": False,
            "is_outbreak_probability": False,
            "is_forecast": False,
            "describes": "observed weather conditions up to the climate provider's data edge",
        },
        "fitted_against": {
            "target_kind": TARGET_KIND,
            "target_statement": TARGET_STATEMENT,
            "disease_group": group,
            "rows": len(rows),
            "events": sum(targets),
            "event_rate": round(sum(targets) / len(rows), 6),
            "panel_start": PANEL_START.isoformat(),
            "panel_end": PANEL_END.isoformat(),
            "districts": len(per_district_week),
            "catalogue_sha256": target_panel.dataset_sha256,
        },
        "fit": {
            "estimator": "l2_logistic_regression_irls",
            "l2": l2,
            "converged": model.converged,
            "iterations": model.iterations,
            "seasonal_baseline_covariate": "seasonal_climatology_logit",
            "coefficients": [round(value, 8) for value in coefficients],
            "means": [round(value, 8) for value in means],
            "deviations": [round(value, 8) for value in deviations],
            "intercept_not_used_for_scoring": round(model.coefficients[0], 8),
            "seasonal_baseline_coefficient_not_used_for_scoring": round(
                model.coefficients[width + 1], 8
            ),
            "note": (
                "Only the environmental coefficients are used to score current "
                "conditions. The intercept and the climatology offset are "
                "deliberately excluded so the experimental scalar carries weather "
                "information alone."
            ),
        },
        "features": {
            "names": list(ENVIRONMENT_FEATURE_NAMES),
            "notes": {
                name: FEATURE_NOTES[name]
                for name in ENVIRONMENT_FEATURE_NAMES
                if name in FEATURE_NOTES
            },
            "full_model_width_for_reference": variant_width,
        },
        "percentile_reference": {
            "definition": (
                "per district and ISO week, the distribution of the environmental "
                f"linear predictor over a +/-{SEASONAL_HALF_WIDTH}-week seasonal "
                "window across every modelled year"
            ),
            "seasonal_half_width_weeks": SEASONAL_HALF_WIDTH,
            "quantile_grid": list(QUANTILE_GRID),
            "minimum_samples": MINIMUM_REFERENCE_SAMPLES,
            "encoding": "[sample_count, q000, q005, ..., q100]",
        },
        "reference_distribution": reference,
        "bands": {label: ceiling for ceiling, label in BANDS},
        "warnings": [
            "This scalar describes weather, not disease.",
            (
                "The historical association it encodes is weak, and that is "
                "measured rather than asserted. A three-way ablation on identical "
                "rows, origins and baselines "
                "(data/forecasting/environment_block_ablation.json) found that "
                "removing the environmental block entirely moves the pooled Brier "
                "score by order 1e-05 against a Brier of order 1e-02, in both "
                "directions depending on horizon. Substantially all of the "
                "catalogue-row model's measured skill comes from seasonal "
                "climatology and row history, not from weather."
            ),
            (
                "So a high percentile means this weak fitted weather index is "
                "unusual for the season in this district. It does not mean an "
                "outbreak is expected and must never be rendered as disease risk "
                "or transmission suitability."
            ),
        ],
    }


def write_model(payload: dict[str, Any], path: Path | None = None) -> Path:
    target = path or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Fit the environmental suitability model.")
    parser.add_argument("--group", default="any_reported_outbreak")
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=ARTEFACT_PATH)
    args = parser.parse_args(argv)
    payload = fit_suitability_model(
        group=args.group,
        l2=args.l2,
        progress=lambda message: print(f"  ... {message}", file=sys.stderr, flush=True),
    )
    path = write_model(payload, args.output)
    fit = payload["fit"]
    names = payload["features"]["names"]
    pairs = sorted(
        zip(names, fit["coefficients"], strict=True), key=lambda item: abs(item[1]), reverse=True
    )
    print(f"rows={payload['fitted_against']['rows']} events={payload['fitted_against']['events']}")
    print("standardised environmental coefficients (largest first):")
    for name, value in pairs:
        print(f"  {name:26} {value:+.5f}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ARTEFACT_PATH",
    "MODEL_VERSION",
    "QUANTITY",
    "QUANTITY_STATEMENT",
    "SCHEMA_VERSION",
    "SuitabilityArtefactInvalid",
    "SuitabilityArtefactMissing",
    "SuitabilityModel",
    "SuitabilityScore",
    "band_for",
    "fit_suitability_model",
    "load_model",
    "write_model",
]
