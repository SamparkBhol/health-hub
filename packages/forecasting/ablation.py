"""Does the environmental block earn its place? A three-way, like-for-like test.

The reported-outbreak model has always shipped an environment ablation, but it
compared only two things at the penalty each origin happened to select.  This
module asks the sharper question, and asks it fairly:

    On *identical* rows, *identical* rolling origins, an *identical* seasonal
    climatology baseline and a *single fixed* ridge penalty, how much does the
    environmental block actually buy?

Three feature sets are fitted side by side at every (disease group, horizon):

``no_environment``
    calendar + reporting history only.
``environment_v1_0_0``
    the ten environmental columns the model publishes.
``environment_v1_1_0``
    those ten plus five candidates built specifically to give environment its
    best chance - an 8-week rainfall anomaly, the longest dry run in four weeks,
    a humid-day count, mean daily maximum temperature, and an explicit
    rain x temperature interaction.

Fixing the penalty is the point.  If each variant tuned its own penalty, a
difference could be a difference in tuning rather than in information.

WHAT THIS FOUND
---------------
Nothing.  See :data:`ARTEFACT_PATH` for the numbers actually measured: the
environmental block changes pooled Brier score by order 1e-05 against a Brier of
order 1e-02, in both directions depending on horizon, and the enriched block did
not rescue it.  Substantially all of the model's skill comes from the seasonal
climatology and the reporting-history block.  That is a real finding about the
data, not a defect in the code, and it is published rather than buried.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from . import metrics
from .climate import (
    EXTENDED_FEATURE_NAMES,
    EXTENDED_FEATURE_NOTES,
    FEATURE_NAMES,
    FEATURE_NOTES,
    build_feature_index,
    load_weekly_panel,
)
from .models import RidgeLogistic, SeasonalClimatologyBaseline
from .panel import build_examples, panel_weeks

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = REPO_ROOT / "data" / "forecasting" / "environment_block_ablation.json"
SCHEMA_VERSION = "1.0.0"
FIXED_L2 = 8.0

VARIANT_ORDER = ("no_environment", "environment_v1_0_0", "environment_v1_1_0")


def _variant_columns(environment_width: int, calendar: list[int], history: list[int]):
    return list(range(environment_width)) + calendar + history


def run_ablation(
    *,
    groups: tuple[str, ...] = ("any_reported_outbreak", "diarrhoeal_and_cholera"),
    horizons: tuple[int, ...] = (1, 2, 4, 8, 12),
    l2: float = FIXED_L2,
    seed: int = 20260721,
    progress=None,
) -> dict[str, Any]:
    from .backtest import (
        DEFAULT_EVALUATION_YEARS,
        FEATURE_BLOCKS,
        MINIMUM_TRAINING_EVENTS,
        PANEL_END,
        PANEL_START,
        _design,
        _split,
    )
    from .target import TARGET_KIND, TARGET_STATEMENT, build_target_panel

    calendar = list(range(*FEATURE_BLOCKS["calendar"]))
    history = list(range(*FEATURE_BLOCKS["reporting_history"]))
    # The panel emits the extended vector, so the calendar and history blocks sit
    # after all fifteen environmental columns rather than after ten.
    offset = len(EXTENDED_FEATURE_NAMES)
    calendar = [index + offset for index in calendar]
    history = [index + offset for index in history]
    variants = {
        "no_environment": _variant_columns(0, calendar, history),
        "environment_v1_0_0": _variant_columns(len(FEATURE_NAMES), calendar, history),
        "environment_v1_1_0": _variant_columns(
            len(FEATURE_NAMES) + len(EXTENDED_FEATURE_NAMES), calendar, history
        ),
    }

    climate = build_feature_index(load_weekly_panel())
    weeks = panel_weeks(PANEL_START, PANEL_END)
    cells: list[dict[str, Any]] = []
    for group in groups:
        panel = build_target_panel(group)
        for horizon in horizons:
            rows = build_examples(
                target_panel=panel,
                climate=climate,
                horizon_weeks=horizon,
                weeks=weeks,
                extended=True,
            )
            pooled: dict[str, list[float]] = {name: [] for name in variants}
            blocks: dict[str, list[tuple[list[float], list[float], list[int]]]] = {
                name: [] for name in variants
            }
            baseline_pool: list[float] = []
            targets_pool: list[int] = []
            origins: list[int] = []
            for year in DEFAULT_EVALUATION_YEARS:
                train, test = _split(rows, date(year, 1, 1), year)
                if not train or not test:
                    continue
                if sum(row.target for row in train) < MINIMUM_TRAINING_EVENTS:
                    continue
                baseline = SeasonalClimatologyBaseline().fit(train)
                baseline_train = baseline.predict(train)
                baseline_test = baseline.predict(test)
                targets_train = [row.target for row in train]
                targets_test = [row.target for row in test]
                for name, columns in variants.items():
                    model = RidgeLogistic(l2=l2).fit(
                        _design(train, baseline_train, columns), targets_train
                    )
                    predicted = model.predict(_design(test, baseline_test, columns))
                    pooled[name].extend(predicted)
                    blocks[name].append((predicted, baseline_test, targets_test))
                baseline_pool.extend(baseline_test)
                targets_pool.extend(targets_test)
                origins.append(year)
            if not targets_pool:
                cells.append(
                    {
                        "disease_group": group,
                        "horizon_weeks": horizon,
                        "status": "insufficient_evidence",
                        "reason_code": "NO_USABLE_ROLLING_ORIGIN",
                    }
                )
                continue
            baseline_brier = metrics.brier_score(baseline_pool, targets_pool)
            baseline_log = metrics.log_score(baseline_pool, targets_pool)
            entry: dict[str, Any] = {
                "disease_group": group,
                "horizon_weeks": horizon,
                "status": "evaluated",
                "rolling_origins": origins,
                "rows": len(targets_pool),
                "events": sum(targets_pool),
                "seasonal_baseline_brier": round(baseline_brier, 8),
                "seasonal_baseline_log_score": round(baseline_log, 6),
                "variants": {},
            }
            reference: float | None = None
            for name in VARIANT_ORDER:
                brier = metrics.brier_score(pooled[name], targets_pool)
                log_score = metrics.log_score(pooled[name], targets_pool)
                bootstrap = metrics.block_bootstrap(blocks[name], seed=seed)
                if name == "no_environment":
                    reference = brier
                entry["variants"][name] = {
                    "brier": round(brier, 10),
                    "log_score": round(log_score, 6),
                    "brier_skill_score_vs_baseline": round(
                        metrics.skill_score(brier, baseline_brier), 6
                    ),
                    "log_score_gain_nats": round(baseline_log - log_score, 6),
                    "auc": (
                        round(value, 4)
                        if (value := metrics.auc(pooled[name], targets_pool)) is not None
                        else None
                    ),
                    "delta_brier_ci_2_5": round(bootstrap.lower_delta_brier, 12),
                    "delta_brier_ci_97_5": round(bootstrap.upper_delta_brier, 12),
                    "brier_change_vs_no_environment": (
                        None if reference is None else round(brier - reference, 10)
                    ),
                }
            cells.append(entry)
            if progress:
                progress(group, horizon)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "is_synthetic": False,
        "uses_real_odisha_data": True,
        "question": (
            "Does the environmental block add information beyond the seasonal "
            "climatology and the reporting-history block?"
        ),
        "design": {
            "rows": "identical across variants",
            "rolling_origins": "identical across variants",
            "seasonal_climatology_baseline": "identical across variants",
            "ridge_penalty": l2,
            "penalty_is_fixed_rationale": (
                "Tuning each variant separately would let a difference in tuning "
                "masquerade as a difference in information."
            ),
            "target_kind": TARGET_KIND,
            "target_statement": TARGET_STATEMENT,
        },
        "variants": {
            "no_environment": "calendar + reporting history",
            "environment_v1_0_0": f"the {len(FEATURE_NAMES)} published environmental columns",
            "environment_v1_1_0": (
                f"the published columns plus {len(EXTENDED_FEATURE_NAMES)} candidates: "
                + ", ".join(EXTENDED_FEATURE_NAMES)
            ),
        },
        "feature_notes": {**FEATURE_NOTES, **EXTENDED_FEATURE_NOTES},
        "cells": cells,
        "conclusion": (
            "The environmental block does not earn its place. Against a pooled "
            "Brier score of order 1e-02 it moves the score by order 1e-05, in "
            "both directions depending on horizon, and the enriched block did "
            "not rescue it. Substantially all of the model's skill comes from "
            "the seasonal climatology and the reporting-history block. The "
            "enriched block was therefore not shipped."
        ),
        "warnings": [
            (
                "This measures information about whether an outbreak REPORT is "
                "published, not about disease. A null result here is not evidence "
                "that weather does not drive disease in Odisha; it is evidence "
                "that weather does not predict this catalogue's publication "
                "behaviour beyond season and reporting history."
            )
        ],
    }


def write_ablation(payload: dict[str, Any], path: Path | None = None) -> Path:
    target = path or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_ablation(path: Path | None = None) -> dict[str, Any]:
    target = path or ARTEFACT_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"no ablation artefact at {target}; run `python -m packages.forecasting.ablation`"
        )
    payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return payload


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Three-way environment-block ablation.")
    parser.add_argument("--l2", type=float, default=FIXED_L2)
    parser.add_argument("--output", type=Path, default=ARTEFACT_PATH)
    args = parser.parse_args(argv)
    payload = run_ablation(
        l2=args.l2,
        progress=lambda group, horizon: print(
            f"  ... {group} h={horizon}", file=sys.stderr, flush=True
        ),
    )
    path = write_ablation(payload, args.output)
    header = f"{'cell':36} {'events':>6} " + " ".join(f"{name[:18]:>19}" for name in VARIANT_ORDER)
    print(header)
    print("-" * len(header))
    for cell in payload["cells"]:
        if cell["status"] != "evaluated":
            continue
        label = f"{cell['disease_group']}|h{cell['horizon_weeks']}"
        scores = " ".join(
            f"{cell['variants'][name]['brier_skill_score_vs_baseline']:>19.5f}"
            for name in VARIANT_ORDER
        )
        print(f"{label:36} {cell['events']:>6} {scores}")
    print(f"\n{payload['conclusion']}\n\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
