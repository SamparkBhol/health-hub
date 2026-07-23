"""Tests for the experimental EpiClim catalogue-row model.

The point of these tests is not that the model is accurate. It is that the
model cannot silently become dishonest: the target stays labelled as membership
in an incomplete frozen file rather than incidence or official publication,
features cannot see their target, and a failed cell cannot leak a probability.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from packages.forecasting import metrics
from packages.forecasting.backtest import (
    ARTEFACT_PATH,
    FEATURE_BLOCKS,
    MINIMUM_EVALUATION_EVENTS,
    PRIMARY_VARIANT,
    evaluate_horizon,
    variant_columns,
    variant_feature_names,
)
from packages.forecasting.climate import (
    MIN_PRIOR_YEARS,
    DistrictClimateFeatures,
    WeeklyClimate,
    weekly_from_receipt,
)
from packages.forecasting.models import (
    GradientBoostedTrees,
    RidgeLogistic,
    SeasonalClimatologyBaseline,
    logit,
    sigmoid,
)
from packages.forecasting.panel import (
    PANEL_FEATURE_NAMES,
    Example,
    build_examples,
    panel_weeks,
)
from packages.forecasting.service import (
    ForecastArtefactInvalid,
    current_week_refusal,
    evaluation,
    load_report,
    probability_map,
    summary,
)
from packages.forecasting.target import (
    DISEASE_GROUPS,
    EPICLIM_STRING_OVERLAY,
    TARGET_STATEMENT,
    TargetDataError,
    TargetPanel,
    build_target_panel,
    load_alias_index,
    week_start,
)
from pipelines.environmental.districts import load_district_points
from pipelines.environmental.models import EnvironmentalValue

CATALOGUE = Path(__file__).resolve().parents[1] / "data" / "epiclim" / "Final_data.csv"
requires_catalogue = pytest.mark.skipif(
    not CATALOGUE.exists(), reason="EpiClim catalogue not downloaded"
)
requires_artefact = pytest.mark.skipif(
    not ARTEFACT_PATH.exists(), reason="real-data forecast artefact not built"
)


# --------------------------------------------------------------------------
# target definition
# --------------------------------------------------------------------------


def test_target_statement_refuses_the_incidence_reading() -> None:
    lowered = TARGET_STATEMENT.lower()
    assert "not incidence" in lowered
    assert "not a case count" in lowered
    assert "frozen epiclim" in lowered
    assert "not an official-report probability" in lowered


@requires_catalogue
def test_every_odisha_report_resolves_to_a_canonical_district() -> None:
    panel = build_target_panel("any_reported_outbreak")
    assert panel.resolution["odisha_rows"] == 358
    assert panel.resolution["resolved_rows"] == 358
    assert panel.resolution["unparseable_date_rows"] == 0
    assert panel.resolution["distinct_district_ids"] == 30
    known = {point.district_id for point in load_district_points()}
    assert {event.district_id for event in panel.events} <= known


@requires_catalogue
def test_disease_groups_partition_the_catalogue() -> None:
    every = build_target_panel("any_reported_outbreak")
    water = build_target_panel("diarrhoeal_and_cholera")
    vector = build_target_panel("vector_borne")
    assert len(water.events) + len(vector.events) == len(every.events)
    assert water.positive_count >= vector.positive_count
    # The catalogue simply does not carry enough vector-borne Odisha reports.
    assert vector.positive_count < MINIMUM_EVALUATION_EVENTS


@requires_catalogue
def test_positive_only_panel_distinguishes_rows_from_file_absence() -> None:
    panel = build_target_panel("any_reported_outbreak")
    assert panel.positive_count == len(panel.district_weeks)
    assert all(panel.observed(district, week) == 1 for district, week in panel.district_weeks)
    # An arbitrary district-week is absent from this file, not known disease-free.
    missing = (sorted(panel.district_weeks)[0][0], date(1999, 1, 4))
    assert missing not in panel.district_weeks
    assert panel.observed(*missing) == 0


def test_unknown_district_string_fails_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "catalogue.csv"
    dataset.write_text(
        ",week_of_outbreak,state_ut,district,Disease,Cases,Deaths,day,mon,year,"
        "Latitude,Longitude,preci,LAI,Temp\n"
        "0,1st week,Odisha,Atlantis,Cholera,10,,2,1,2022,20.0,85.0,0.1,3,300\n",
        encoding="utf-8",
    )
    with pytest.raises(TargetDataError, match="unresolved"):
        build_target_panel("any_reported_outbreak", dataset=dataset, verify_digest=False)


def test_altered_catalogue_is_refused(tmp_path: Path) -> None:
    dataset = tmp_path / "catalogue.csv"
    dataset.write_text("not the audited file", encoding="utf-8")
    with pytest.raises(TargetDataError, match="digest"):
        build_target_panel("any_reported_outbreak", dataset=dataset)


def test_overlay_rules_are_explicit_and_documented() -> None:
    index = load_alias_index()
    for raw, (district_id, reason) in EPICLIM_STRING_OVERLAY.items():
        assert index["".join(c for c in raw if c.isalnum())] == district_id
        assert len(reason) > 20


def test_unknown_disease_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown disease group"):
        build_target_panel("influenza_h5n1")


# --------------------------------------------------------------------------
# environmental features
# --------------------------------------------------------------------------


@dataclass
class _Receipt:
    values: tuple


def _daily(start: date, days: int, rain: float = 1.0) -> _Receipt:
    values = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        for parameter, value in (
            ("PRECTOTCORR", rain),
            ("T2M", 27.0),
            ("T2M_MAX", 32.0),
            ("T2M_MIN", 22.0),
            ("RH2M", 80.0),
        ):
            values.append(EnvironmentalValue(day=day, parameter=parameter, value=value, unit="x"))
    return _Receipt(values=tuple(values))


def test_weekly_aggregation_drops_partial_weeks() -> None:
    # Start mid-week so the first ISO week is incomplete.
    weeks = weekly_from_receipt(_daily(date(2020, 1, 2), 21, rain=2.0))
    assert all(item.days == 7 for item in weeks if item.complete)
    complete = [item for item in weeks if item.complete]
    assert complete
    assert complete[0].rain_mm == pytest.approx(14.0)
    assert complete[0].t2m_c == pytest.approx(27.0)
    assert complete[0].week.weekday() == 0


def _synthetic_weeks(years: int, *, rain=lambda index: 10.0) -> tuple[WeeklyClimate, ...]:
    start = week_start(date(2008, 1, 7))
    return tuple(
        WeeklyClimate(
            week=start + timedelta(weeks=index),
            days=7,
            rain_mm=rain(index),
            t2m_c=27.0 + math.sin(index / 8.0),
            tmax_c=32.0,
            tmin_c=22.0,
            rh_pct=75.0 + math.cos(index / 8.0),
        )
        for index in range(52 * years)
    )


def test_anomaly_requires_several_prior_years() -> None:
    features = DistrictClimateFeatures(_synthetic_weeks(MIN_PRIOR_YEARS + 2))
    weeks = [item.week for item in features.weeks]
    early = features.features(weeks[10])
    assert early is None, "an anomaly cannot be formed without enough prior years"
    late = features.features(weeks[-1])
    assert late is not None
    assert len(late) == FEATURE_BLOCKS["environment"][1]


def test_trailing_windows_refuse_to_bridge_a_dropped_week() -> None:
    weeks = _synthetic_weeks(6)
    gapped = weeks[:200] + weeks[201:]
    probe = weeks[203].week
    assert DistrictClimateFeatures(weeks).features(probe) is not None
    assert DistrictClimateFeatures(gapped).features(probe) is None


def test_environmental_features_never_read_the_future() -> None:
    """Changing a future week must not change a past feature vector."""

    baseline = _synthetic_weeks(6)
    index = len(baseline) - 20
    probe = baseline[index - 1].week
    before = DistrictClimateFeatures(baseline).features(probe)
    mutated = list(baseline)
    for offset in range(index, len(mutated)):
        item = mutated[offset]
        mutated[offset] = WeeklyClimate(
            week=item.week,
            days=7,
            rain_mm=item.rain_mm + 500.0,
            t2m_c=item.t2m_c + 9.0,
            tmax_c=item.tmax_c,
            tmin_c=item.tmin_c,
            rh_pct=item.rh_pct + 9.0,
        )
    after = DistrictClimateFeatures(tuple(mutated)).features(probe)
    assert before == after


# --------------------------------------------------------------------------
# panel construction
# --------------------------------------------------------------------------


def _panel_fixture(event_weeks: set[int], weeks: list[date]) -> TargetPanel:
    return TargetPanel(
        group="test",
        diseases=("Test",),
        events=(),
        district_weeks=frozenset(("D1", weeks[index]) for index in event_weeks),
        dataset_sha256="0" * 64,
        resolution={},
    )


def _climate_fixture() -> dict[str, DistrictClimateFeatures]:
    return {"D1": DistrictClimateFeatures(_synthetic_weeks(8))}


def test_panel_targets_are_exactly_horizon_weeks_ahead() -> None:
    climate = _climate_fixture()
    weeks = [item.week for item in climate["D1"].weeks]
    rows = build_examples(
        target_panel=_panel_fixture({200}, weeks),
        climate=climate,
        horizon_weeks=4,
        weeks=weeks,
        history_start=weeks[0],
    )
    assert rows
    for row in rows:
        assert (row.target_week - row.issue_week).days == 28
    positives = [row for row in rows if row.target == 1]
    assert [row.issue_week for row in positives] == [weeks[196]]


def test_reporting_history_respects_the_publication_lag() -> None:
    climate = _climate_fixture()
    weeks = [item.week for item in climate["D1"].weeks]
    event_index = 200
    rows = build_examples(
        target_panel=_panel_fixture({event_index}, weeks),
        climate=climate,
        horizon_weeks=1,
        weeks=weeks,
        report_lag_weeks=2,
        history_start=weeks[0],
    )
    column = PANEL_FEATURE_NAMES.index("district_reports_4w")
    by_issue = {row.issue_week: row for row in rows}
    # The report becomes usable only two weeks after the event week.
    assert by_issue[weeks[event_index]].features[column] == 0.0
    assert by_issue[weeks[event_index + 1]].features[column] == 0.0
    assert by_issue[weeks[event_index + 2]].features[column] == 1.0
    since_column = PANEL_FEATURE_NAMES.index("log1p_weeks_since_district_report")
    # At the first issue where the sensitivity lag makes the row usable, it is
    # zero weeks old. This catches an ordering bug that previously reported one.
    assert by_issue[weeks[event_index + 2]].features[since_column] == pytest.approx(0.0)


def test_panel_feature_names_match_the_vector_width() -> None:
    climate = _climate_fixture()
    weeks = [item.week for item in climate["D1"].weeks]
    rows = build_examples(
        target_panel=_panel_fixture({150}, weeks),
        climate=climate,
        horizon_weeks=1,
        weeks=weeks,
        history_start=weeks[0],
    )
    assert len(rows[0].features) == len(PANEL_FEATURE_NAMES)
    assert variant_feature_names(PRIMARY_VARIANT)[-1] == "seasonal_baseline_logit"
    assert len(variant_columns(PRIMARY_VARIANT)) == len(PANEL_FEATURE_NAMES)


def test_panel_weeks_are_contiguous_mondays() -> None:
    weeks = panel_weeks(date(2011, 1, 1), date(2011, 12, 31))
    assert all(item.weekday() == 0 for item in weeks)
    assert all((weeks[index + 1] - weeks[index]).days == 7 for index in range(len(weeks) - 1))


# --------------------------------------------------------------------------
# model ladder
# --------------------------------------------------------------------------


def _rows(seed: int, count: int = 4000) -> list[Example]:
    generator = random.Random(seed)  # noqa: S311 - deterministic test fixture, not security material
    rows: list[Example] = []
    start = date(2015, 1, 5)
    for index in range(count):
        week = start + timedelta(weeks=index // 4)
        driver = generator.gauss(0.0, 1.0)
        probability = sigmoid(-3.0 + 1.4 * driver)
        rows.append(
            Example(
                district_id=f"D{index % 4}",
                issue_week=week,
                target_week=week + timedelta(weeks=1),
                target_week_of_year=week.isocalendar().week,
                features=(driver, generator.gauss(0.0, 1.0)),
                target=int(generator.random() < probability),
            )
        )
    return rows


def test_ridge_logistic_recovers_a_known_signal() -> None:
    rows = _rows(11)
    matrix = [list(row.features) for row in rows]
    targets = [row.target for row in rows]
    model = RidgeLogistic(l2=1.0).fit(matrix, targets)
    assert model.converged
    assert model.coefficients[1] > 0.5, "the true driver must get a positive weight"
    assert abs(model.coefficients[2]) < 0.3, "the noise column must stay small"
    predictions = model.predict(matrix)
    assert 0.0 < min(predictions) <= max(predictions) < 1.0


def test_stronger_penalty_shrinks_coefficients() -> None:
    rows = _rows(12)
    matrix = [list(row.features) for row in rows]
    targets = [row.target for row in rows]
    weak = RidgeLogistic(l2=0.5).fit(matrix, targets)
    strong = RidgeLogistic(l2=5000.0).fit(matrix, targets)
    assert abs(strong.coefficients[1]) < abs(weak.coefficients[1])


def test_seasonal_baseline_is_calibrated_on_its_own_training_data() -> None:
    rows = _rows(13)
    baseline = SeasonalClimatologyBaseline().fit(rows)
    predictions = baseline.predict(rows)
    observed = sum(row.target for row in rows) / len(rows)
    assert abs(sum(predictions) / len(predictions) - observed) < 0.02
    assert all(0.0 < value < 1.0 for value in predictions)


def test_empty_training_sets_are_refused() -> None:
    with pytest.raises(ValueError):
        SeasonalClimatologyBaseline().fit([])
    with pytest.raises(ValueError):
        RidgeLogistic().fit([], [])
    with pytest.raises(ValueError):
        GradientBoostedTrees().fit([], [])


def test_gradient_boosting_learns_a_monotone_response() -> None:
    rows = _rows(14, count=3000)
    matrix = [list(row.features) for row in rows]
    targets = [row.target for row in rows]
    booster = GradientBoostedTrees(rounds=30).fit(matrix, targets)
    low = booster.predict([[-2.0, 0.0]])[0]
    high = booster.predict([[2.0, 0.0]])[0]
    assert high > low


def test_logit_and_sigmoid_round_trip() -> None:
    for value in (0.001, 0.05, 0.5, 0.9):
        assert sigmoid(logit(value)) == pytest.approx(value, abs=1e-6)


# --------------------------------------------------------------------------
# scoring rules and calibration
# --------------------------------------------------------------------------


def test_brier_and_log_score_have_known_values() -> None:
    assert metrics.brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert metrics.brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert metrics.log_score([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert metrics.skill_score(0.05, 0.10) == pytest.approx(0.5)


def test_proper_scores_reward_the_truth() -> None:
    generator = random.Random(5)  # noqa: S311 - deterministic test fixture, not security material
    truth = [generator.random() * 0.4 for _ in range(4000)]
    targets = [int(generator.random() < value) for value in truth]
    honest = metrics.log_score(truth, targets)
    for distortion in (0.5, 1.5, 2.5):
        skewed = [min(0.999, value * distortion) for value in truth]
        assert metrics.log_score(skewed, targets) > honest


def test_reliability_bins_cover_every_row() -> None:
    generator = random.Random(6)  # noqa: S311 - deterministic test fixture, not security material
    probabilities = [generator.random() * 0.2 for _ in range(1000)]
    targets = [int(generator.random() < value) for value in probabilities]
    curve = metrics.reliability_bins(probabilities, targets, bins=5)
    assert sum(int(item["count"]) for item in curve) == 1000
    assert metrics.expected_calibration_error(probabilities, targets) < 0.05


def test_randomised_pit_is_flat_for_a_calibrated_forecaster() -> None:
    generator = random.Random(7)  # noqa: S311 - deterministic test fixture, not security material
    probabilities = [generator.random() * 0.3 for _ in range(20000)]
    targets = [int(generator.random() < value) for value in probabilities]
    report = metrics.randomised_pit(probabilities, targets)
    assert report["sample_size"] == 20000
    assert sum(report["histogram"]) == 20000
    assert float(report["uniformity_chi_square"]) < 30.0


def test_block_bootstrap_is_deterministic_and_widens_with_disagreement() -> None:
    blocks = [
        # Season one favours the model, season two favours the reference.
        ([0.1] * 50, [0.2] * 50, [0] * 50),
        ([0.2] * 50, [0.1] * 50, [0] * 50),
    ]
    first = metrics.block_bootstrap(blocks, replicates=200, seed=3)
    second = metrics.block_bootstrap(blocks, replicates=200, seed=3)
    assert first.as_dict() == second.as_dict()
    assert first.lower_delta_brier < 0 < first.upper_delta_brier


def test_auc_is_half_for_noise_and_one_for_perfect_ranking() -> None:
    assert metrics.auc([0.1, 0.9], [0, 1]) == pytest.approx(1.0)
    assert metrics.auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert metrics.auc([0.1, 0.2], [0, 0]) is None


# --------------------------------------------------------------------------
# publication gate
# --------------------------------------------------------------------------


def _block_width(name: str) -> int:
    start, end = FEATURE_BLOCKS[name]
    return end - start


def _gate_rows(seed: int, *, informative: bool) -> list[Example]:
    """Four districts of weekly rows where the driver is real or pure noise."""

    generator = random.Random(seed)  # noqa: S311 - deterministic test fixture, not security material
    rows: list[Example] = []
    for district in range(4):
        week = date(2011, 1, 3)
        previous = 0
        for index in range(52 * 12):
            driver = generator.gauss(0.0, 1.0)
            seasonal = math.sin(2 * math.pi * (index % 52) / 52.0)
            linear = -3.4 + 0.8 * seasonal + (1.6 * driver if informative else 0.0)
            target = int(generator.random() < sigmoid(linear))
            rows.append(
                Example(
                    district_id=f"D{district}",
                    issue_week=week,
                    target_week=week + timedelta(weeks=1),
                    target_week_of_year=(week + timedelta(weeks=1)).isocalendar().week,
                    # Only lagged outcomes may appear as features; putting the row's
                    # own target here would be exactly the leak the design forbids.
                    # Widths are read from FEATURE_BLOCKS so that adding an
                    # environmental column cannot silently break this fixture.
                    features=(
                        (driver,) * _block_width("environment")
                        + (seasonal,) * _block_width("calendar")
                        + (float(previous),) * _block_width("reporting_history")
                    ),
                    target=target,
                )
            )
            previous = target
            week += timedelta(weeks=1)
    return rows


def test_gate_retains_a_genuinely_informative_experimental_cell() -> None:
    report = evaluate_horizon(
        _gate_rows(21, informative=True),
        horizon_weeks=1,
        evaluation_years=(2019, 2020, 2021, 2022),
        challenger=False,
        l2=4.0,
        seed=11,
    )
    assert report["status"] == "experimental"
    assert report["reason_codes"] == []
    assert report["evaluation"]["model_brier"] < report["evaluation"]["seasonal_baseline_brier"]
    assert report["season_block_bootstrap"]["delta_brier_ci_2_5"] > 0


def test_gate_refuses_a_cell_with_no_real_signal() -> None:
    report = evaluate_horizon(
        _gate_rows(22, informative=False),
        horizon_weeks=1,
        evaluation_years=(2019, 2020, 2021, 2022),
        challenger=False,
        l2=4.0,
        seed=11,
    )
    assert report["status"] == "insufficient_evidence"
    assert report["reason_codes"]


def test_gate_refuses_when_no_origin_has_enough_training_events() -> None:
    rows = [
        Example(
            district_id="D0",
            issue_week=date(2021, 1, 4) + timedelta(weeks=index),
            target_week=date(2021, 1, 11) + timedelta(weeks=index),
            target_week_of_year=(date(2021, 1, 11) + timedelta(weeks=index)).isocalendar().week,
            features=(0.0,) * len(PANEL_FEATURE_NAMES),
            target=0,
        )
        for index in range(60)
    ]
    report = evaluate_horizon(
        rows,
        horizon_weeks=1,
        evaluation_years=(2022,),
        challenger=False,
        l2=1.0,
        seed=1,
    )
    assert report["status"] == "insufficient_evidence"
    assert report["reason_codes"] == ["NO_USABLE_ROLLING_ORIGIN"]
    assert "evaluation" not in report


# --------------------------------------------------------------------------
# published artefact and read-side service
# --------------------------------------------------------------------------


def _artefact(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "model_version": "test",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "is_synthetic": False,
        "target": {
            "kind": "experimental_epiclim_catalogue_row_occurrence",
            "is_incidence": False,
            "is_case_count": False,
            "experimental": True,
            "is_official_publication_probability": False,
            "is_operational_forecast": False,
        },
        "protocol": {},
        "models": {},
        "data": {"panel": {"end": "2022-12-31"}},
        "results": [
            {
                "disease_group": "demo",
                "status": "evaluated",
                "horizons": [
                    {
                        "horizon_weeks": 1,
                        "status": "experimental",
                        "reason_codes": [],
                        "evaluation": {
                            "model_brier": 0.01,
                            "seasonal_baseline_brier": 0.02,
                            "brier_skill_score_vs_baseline": 0.5,
                            "log_score_gain_nats": 0.01,
                        },
                        "calibration": {},
                        "season_block_bootstrap": {},
                        "latest_issue_map": {
                            "status": "historical_reissue",
                            "issue_week": "2022-12-19",
                            "target_week": "2022-12-26",
                            "districts": [{"district_id": "OD-DIST-puri"}],
                        },
                    },
                    {
                        "horizon_weeks": 4,
                        "status": "insufficient_evidence",
                        "reason_codes": ["DOES_NOT_BEAT_SEASONAL_CLIMATOLOGY_BRIER"],
                        "evaluation": {
                            "model_brier": 0.03,
                            "seasonal_baseline_brier": 0.02,
                            "brier_skill_score_vs_baseline": -0.5,
                            "log_score_gain_nats": -0.01,
                        },
                    },
                ],
            }
        ],
        "experimental_cells": [{"disease_group": "demo", "horizon_weeks": 1}],
        "published_cells": [],
        "refused_cells": [{"disease_group": "demo", "horizon_weeks": 4}],
    }
    payload.update(overrides)
    path = tmp_path / "artefact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_service_refuses_an_artefact_that_claims_to_be_synthetic(tmp_path: Path) -> None:
    with pytest.raises(ForecastArtefactInvalid):
        load_report(_artefact(tmp_path, is_synthetic=True))


def test_service_refuses_an_artefact_that_claims_incidence(tmp_path: Path) -> None:
    with pytest.raises(ForecastArtefactInvalid):
        load_report(_artefact(tmp_path, target={"is_incidence": True, "is_case_count": False}))


def test_refused_cell_returns_no_probabilities(tmp_path: Path) -> None:
    report = load_report(_artefact(tmp_path))
    payload = probability_map("demo", 4, report)
    assert payload["status"] == "insufficient_evidence"
    assert payload["capability_code"] == "insufficient_evidence"
    assert payload["districts"] == []
    assert payload["reason_codes"] == ["DOES_NOT_BEAT_SEASONAL_CLIMATOLOGY_BRIER"]
    assert payload["is_incidence"] is False


def test_unknown_cell_returns_a_typed_refusal(tmp_path: Path) -> None:
    report = load_report(_artefact(tmp_path))
    payload = probability_map("demo", 99, report)
    assert payload["status"] == "insufficient_evidence"
    assert payload["reason_codes"] == ["CELL_NOT_EVALUATED"]
    assert evaluation("nope", 1, report)["status"] == "insufficient_evidence"


def test_experimental_cell_is_labelled_as_epiclim_file_membership(tmp_path: Path) -> None:
    report = load_report(_artefact(tmp_path))
    payload = probability_map("demo", 1, report)
    assert payload["status"] == "experimental"
    assert payload["is_synthetic"] is False
    assert payload["is_incidence"] is False
    assert payload["is_case_count"] is False
    assert payload["quantity"] == "experimental_epiclim_catalogue_row_occurrence"
    assert payload["experimental"] is True
    assert payload["is_official_publication_probability"] is False
    assert payload["is_operational_forecast"] is False
    assert payload["districts"]
    assert "not incidence" in payload["warning"].lower()


def test_current_week_is_refused_with_an_unlock_path(tmp_path: Path) -> None:
    report = load_report(_artefact(tmp_path))
    payload = current_week_refusal(date(2026, 7, 21), report)
    assert payload["status"] == "insufficient_evidence"
    assert payload["districts"] == []
    assert payload["target_series_supported_to"] == "2022-12-31"
    assert "IHIP" in payload["unlocked_by"]


def test_summary_lists_every_cell(tmp_path: Path) -> None:
    report = load_report(_artefact(tmp_path))
    payload = summary(report)
    assert {cell["horizon_weeks"] for cell in payload["cells"]} == {1, 4}
    assert payload["is_synthetic"] is False


# --------------------------------------------------------------------------
# the artefact this repository actually ships
# --------------------------------------------------------------------------


@requires_artefact
def test_shipped_artefact_is_internally_consistent() -> None:
    report = load_report()
    assert report["is_synthetic"] is False
    assert report["uses_real_odisha_data"] is True
    assert report["target"]["is_incidence"] is False
    assert report["protocol"]["random_splits_used"] is False
    assert report["data"]["target_catalogue"]["positive_only"] is True
    assert report["data"]["environment"]["districts"] == 30
    groups = {entry["disease_group"] for entry in report["results"]}
    assert set(DISEASE_GROUPS) <= groups


@requires_artefact
def test_shipped_artefact_never_leaks_a_refused_probability() -> None:
    report = load_report()
    for group in report["results"]:
        for horizon in group["horizons"]:
            if horizon["status"] == "experimental":
                assert horizon["reason_codes"] == []
                assert horizon["latest_issue_map"]["districts"]
                assert len(horizon["latest_issue_map"]["districts"]) == 30
            else:
                assert horizon["reason_codes"]
                assert "latest_issue_map" not in horizon


@requires_artefact
def test_shipped_artefact_gate_matches_its_own_numbers() -> None:
    report = load_report()
    for group in report["results"]:
        for horizon in group["horizons"]:
            evaluation_block = horizon.get("evaluation")
            if not evaluation_block:
                continue
            beats_brier = (
                evaluation_block["model_brier"] < evaluation_block["seasonal_baseline_brier"]
            )
            beats_log = evaluation_block["log_score_gain_nats"] > 0
            enough = evaluation_block["events"] >= MINIMUM_EVALUATION_EVENTS
            bootstrap = horizon["season_block_bootstrap"]
            significant = (
                bootstrap["delta_brier_ci_2_5"] > 0 and bootstrap["delta_log_score_ci_2_5"] > 0
            )
            expected = enough and beats_brier and beats_log and significant
            assert (horizon["status"] == "experimental") is expected


@requires_artefact
def test_shipped_artefact_reports_the_environment_ablation() -> None:
    report = load_report()
    for group in report["results"]:
        for horizon in group["horizons"]:
            if "evaluation" not in horizon:
                continue
            ablation = horizon["environment_ablation"]
            assert ablation["variant"] == "reporting_history_only"
            # The gain is reported whatever its sign; hiding a negative gain
            # would be exactly the dishonesty this artefact exists to prevent.
            assert isinstance(ablation["environment_brier_gain"], (int, float))
            assert isinstance(ablation["environment_log_score_gain_nats"], (int, float))


@requires_artefact
def test_shipped_artefact_selects_regularisation_without_touching_evaluation() -> None:
    report = load_report()
    assert (
        report["protocol"]["hyperparameter_selection"]
        == "nested rolling-origin validation inside the training window only"
    )
    for group in report["results"]:
        for horizon in group["horizons"]:
            if "hyperparameter_selection" not in horizon:
                continue
            for entry in horizon["hyperparameter_selection"]:
                origin_year = int(entry["origin"][:4])
                for fold in entry["inner_validation"]:
                    assert fold["inner_season"] < origin_year


@requires_artefact
def test_vector_borne_is_refused_before_any_fit() -> None:
    report = load_report()
    vector = next(entry for entry in report["results"] if entry["disease_group"] == "vector_borne")
    assert vector["status"] == "insufficient_evidence"
    assert vector["reason_codes"] == ["CATALOGUE_EVENTS_BELOW_MINIMUM"]
    assert all(item["status"] == "insufficient_evidence" for item in vector["horizons"])
