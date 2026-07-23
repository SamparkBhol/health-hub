#!/usr/bin/env python3
"""Exercise all three brief objectives end to end and print what came back.

It calls the same functions as the HTTP API, so what it prints is what a reader
gets from the running service -- no recorded fixtures or stand-ins.

    uv run python scripts/sample_walkthrough.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def objective_one() -> None:
    rule("OBJECTIVE 1  Crawl Odia / Hindi / English health information for Odisha")
    from workers.ingestion.registry import load_registry

    sources = load_registry().sources
    enabled = [source for source in sources if source.enabled]
    by_language: dict[str, int] = {}
    for source in enabled:
        for language in source.languages:
            by_language[language] = by_language.get(language, 0) + 1
    print(f"registered routes : {len(sources)}")
    print(f"enabled routes    : {len(enabled)}")
    print(f"by language       : {json.dumps(by_language, sort_keys=True)}")
    for language in ("or", "hi", "en"):
        example = next(
            source for source in enabled if language in source.languages
        )
        print(f"  {language}: {example.id} -> {example.url}")


def objective_two() -> None:
    rule("OBJECTIVE 2  District disease pattern / heatmap across Odisha")
    from services.api.public_health import hmis_map, malaria_map

    annual = malaria_map(year=2024, metric="api")
    rows = sorted(annual["records"], key=lambda row: -(row["value"] or 0.0))
    print(f"layer            : official NCVBDC annual, {annual['year']}")
    print(f"districts        : {len(annual['records'])}  synthetic={annual['is_synthetic']}")
    print("top 5 by API     :")
    for row in rows[:5]:
        print(
            f"  {row['district_name']:<15} API {row['value']:>6}"
            f"   cases {row['total_cases']:>7,}"
        )
    monthly = hmis_map()
    observed = [
        row for row in monthly["records"] if row["observation_state"] == "observed"
    ]
    print(
        f"HMIS month layer : {monthly['period']} "
        f"{len(observed)}/{len(monthly['records'])} districts observed"
    )


def objective_three() -> None:
    rule("OBJECTIVE 3  Predictive analysis of outbreak likelihood")
    from services.api.public_health import public_outlook_evaluation, public_outlook_map

    evaluation = public_outlook_evaluation()
    print(f"training rows    : {evaluation['modeling_rows']:,}")
    print(f"events           : {evaluation['events']:,}")
    print(f"rolling origins  : {', '.join(evaluation['origins'])}")
    print("\nmodel ladder (pooled out-of-time, lower Brier is better):")
    ladder = sorted(
        evaluation["pooled"].items(), key=lambda item: item[1]["brier"]
    )
    for name, scores in ladder:
        marker = "  <- selected" if name == evaluation["selected_by_brier"] else ""
        print(
            f"  {name:<34} brier {scores['brier']:.5f}"
            f"  auc {scores['auc']:.3f}{marker}"
        )
    print(
        f"\nbeats the null constant : {evaluation['beats_unconditional_climatology']}"
        f"  (Brier skill {evaluation['brier_skill_score_vs_unconditional']})"
    )
    ablation = evaluation["environment_block_ablation"]
    print(
        f"environment ablation    : with {ablation['with_environment']['brier']:.5f}"
        f" / without {ablation['without_environment']['brier']:.5f}"
        f" -> delta {ablation['delta_brier']:+.5f}"
    )
    print(f"  per origin            : {ablation['delta_brier_by_origin']}")

    outlook = public_outlook_map(horizon_month=1)
    records = sorted(
        outlook["records"],
        key=lambda row: -float(row["research_indicator_probability"]),
    )
    distinct = len({round(float(r["research_indicator_probability"]), 8) for r in records})
    print(f"\nserved outlook          : {outlook['status']}, horizon 1 month")
    print(f"distinct probabilities  : {distinct}/{len(records)}")
    print("top 5 modelled likelihoods:")
    for row in records[:5]:
        print(
            f"  {row['district_name']:<15}"
            f" {100.0 * float(row['research_indicator_probability']):>5.1f}%"
            f"   rain {row['forecast_precipitation_mean_mm']:>4.0f} mm"
            f"   temp {row['forecast_temperature_mean_c']:.1f} C"
        )
    print("lowest 3:")
    for row in records[-3:]:
        print(
            f"  {row['district_name']:<15}"
            f" {100.0 * float(row['research_indicator_probability']):>5.1f}%"
        )
    print(f"\nskill attribution : {outlook['skill_attribution']}")
    print(f"calibration state : {outlook['forecast_calibration_state']}")


def main() -> int:
    objective_one()
    objective_two()
    objective_three()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
