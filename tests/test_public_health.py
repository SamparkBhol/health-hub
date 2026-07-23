from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from packages.forecasting.public_hmis import (
    FEATURE_NAMES,
    HISTORY_FEATURE_INDICES,
    SEASON_FEATURE_INDICES,
)
from pipelines.environmental.seasonal import ENSEMBLE_MEMBERS, load_seasonal_outlook
from pipelines.surveillance.hmis import load_hmis_rows
from pipelines.surveillance.ncvbdc import load_ncvbdc_rows
from services.api.main import create_app
from services.api.public_health import (
    SERVED_SOURCE_MODEL,
    _blended_month_rate,
    _month_weights,
    hmis_map,
    malaria_map,
    public_outlook_evaluation,
    public_outlook_map,
)


def test_official_ncvbdc_artifact_is_a_complete_annual_odisha_panel() -> None:
    rows = load_ncvbdc_rows()
    assert len(rows) == 15 * 30
    assert {row.year for row in rows} == set(range(2010, 2025))
    for year in range(2010, 2025):
        selected = [row for row in rows if row.year == year]
        assert len(selected) == 30
        assert len({row.district_id for row in selected}) == 30
        assert all(len(row.source_sha256) == 64 for row in selected)

    latest = malaria_map()
    assert latest["year"] == 2024
    assert len(latest["records"]) == 30
    assert sum(row["total_cases"] for row in latest["records"]) == 68_693


def test_hmis_artifact_preserves_missing_values_instead_of_zero_imputation() -> None:
    rows = load_hmis_rows()
    assert len(rows) == 96 * 30
    assert len({row.period_start for row in rows}) == 96
    assert min(row.period_start for row in rows) == "2012-04-01"
    assert max(row.period_start for row in rows) == "2020-03-01"

    dengue = [row.dengue_positive_records for row in rows]
    assert any(value is None for value in dengue)
    assert any(value == 0 for value in dengue if value is not None)
    layer = hmis_map()
    assert len(layer["records"]) == 30
    assert "not deduplicated" in layer["metric_scope"]


def test_seasonal_artifact_contains_three_real_ensemble_windows_per_district() -> None:
    outlook = load_seasonal_outlook()
    assert outlook is not None
    assert outlook["ensemble_members"] == ENSEMBLE_MEMBERS
    assert len(outlook["districts"]) == 30
    assert len({row["district_id"] for row in outlook["districts"]}) == 30
    for district in outlook["districts"]:
        assert len(district["source_sha256"]) == 64
        assert len(district["windows"]) == 3
        assert {window["horizon_month"] for window in district["windows"]} == {1, 2, 3}
        for window in district["windows"]:
            assert window["ensemble_size"] == ENSEMBLE_MEMBERS
            assert (
                window["precipitation_p10_mm"]
                <= window["precipitation_mean_mm"]
                <= window["precipitation_p90_mm"]
            )
            assert (
                window["temperature_p10_c"]
                <= window["temperature_mean_c"]
                <= window["temperature_p90_c"]
            )


def test_public_model_keeps_the_winning_baseline_and_reports_ablation() -> None:
    evaluation = public_outlook_evaluation()
    # 1,936 = 2,026 observable district-months minus the first three per district,
    # which cannot carry a lag block. Losing 90 rows to gain the autocorrelation
    # signal is the trade that turned a zero-skill model into a skilful one.
    assert evaluation["modeling_rows"] == 1_936
    assert evaluation["events"] == 380
    assert len(evaluation["folds"]) == 3
    assert evaluation["origins"] == ["2017-01-01", "2018-01-01", "2019-01-01"]
    briers = {
        name: values["brier"] for name, values in evaluation["pooled"].items()
    }
    assert evaluation["selected_by_brier"] == min(briers, key=briers.get)
    # True since the lagged-disease block entered the design. Left as an equality
    # against the ladder rather than a pinned literal so it tracks the scores.
    assert evaluation["ridge_beats_baseline"] is (
        briers["ridge_logistic"] < briers["calendar_month_baseline"]
    )

    # The null competitor -- a constant equal to the training base rate -- must be
    # in the contest. Without it a model can be crowned having beaten only the
    # baselines it was allowed to face. Naming a specific winner here would make
    # the test reward whichever model happened to win rather than the discipline;
    # what must hold is that the comparison exists and its verdict is published.
    assert "unconditional_climatology" in briers
    assert isinstance(evaluation["beats_unconditional_climatology"], bool)
    assert evaluation["beats_unconditional_climatology"] is (
        evaluation["selected_by_brier"] != "unconditional_climatology"
    )

    # The lagged-disease block is what beats the null competitor. Asserting the
    # skill rather than a model name keeps the test honest if the winner changes.
    assert evaluation["beats_unconditional_climatology"] is True
    assert evaluation["brier_skill_score_vs_unconditional"] > 0.2
    ladder = {name: values["brier"] for name, values in evaluation["pooled"].items()}
    assert ladder[evaluation["selected_by_brier"]] < ladder["persistence_previous_month"]
    # The pre-fix model is kept in the ladder so the gain stays attributable to
    # disease history rather than to retuning.
    assert ladder["season_environment_no_history"] > ladder["unconditional_climatology"]
    # The target's base rate collapses across the series, so the over-forecast
    # ratio must be measured and shipped rather than left for a reader to derive.
    assert evaluation["over_forecast_ratio"] > 0.0
    assert set(evaluation["target_base_rate_by_year"]) >= {"2014", "2020"}

    outlook = public_outlook_map(horizon_month=3)
    assert len(outlook["records"]) == 30
    # The matched ablation removes rainfall and temperature from the *same* model
    # and scores no worse, so the environment block is context, not skill -- even
    # though it rides along in the served vector.
    assert outlook["environment_promoted"] is False
    assert outlook["environment_ablation_result"].startswith("did_not_reduce_brier")
    assert "disease history" in outlook["skill_attribution"]
    assert "not probability" in outlook["priority_definition"]


def test_public_health_api_and_agent_use_the_real_layers(tmp_path) -> None:
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'public-health.sqlite3'}"))

    malaria = client.get("/api/v1/public-health/malaria/map?year=2024&metric=api")
    assert malaria.status_code == 200
    assert len(malaria.json()["data"]["records"]) == 30

    outlook = client.get("/api/v1/outlook/public/map?disease=malaria&horizon_month=3")
    assert outlook.status_code == 200
    assert len(outlook.json()["data"]["records"]) == 30
    assert outlook.json()["data"]["status"] == "research_outlook"

    invalid = client.get("/api/v1/outlook/public/map?disease=dengue&horizon_month=3")
    assert invalid.status_code == 422

    observed_answer = client.post(
        "/api/v1/agent/query",
        json={"question": "How many malaria cases were reported in Kandhamal?"},
    ).json()["data"]
    assert observed_answer["answer_state"] == "official_annual_observation"
    assert observed_answer["observation"]["year"] == 2024
    assert observed_answer["observation"]["records"][0]["district_id"] == (
        "OD-DIST-kandhamal"
    )
    assert observed_answer["is_synthetic"] is False
    assert observed_answer["citations"]

    forecast_answer = client.post(
        "/api/v1/agent/query",
        json={"question": "Predict malaria risk in Kandhamal for the next 3 months"},
    ).json()["data"]
    assert forecast_answer["answer_state"] == "public_research_outlook"
    assert forecast_answer["outlook"]["horizon_month"] == 3
    assert forecast_answer["outlook"]["districts"][0]["district_id"] == (
        "OD-DIST-kandhamal"
    )
    assert forecast_answer["outlook"]["is_synthetic"] is False


def test_environment_block_ablation_is_a_matched_like_for_like_comparison() -> None:
    """"Environment did not help" must be measured, not merely asserted."""

    evaluation = public_outlook_evaluation()
    ablation = evaluation["environment_block_ablation"]

    # Both arms: same rows, same rolling origins, same estimator, same penalty.
    assert "identical rolling origins" in ablation["design"]
    assert "l2=10.0" in ablation["design"]
    assert ablation["with_environment"]["features"] == list(FEATURE_NAMES)
    # The counterfactual must differ from the full model by the environment block
    # and nothing else. Comparing against season-only would drop the disease
    # history at the same time and credit rainfall with history's contribution.
    assert ablation["without_environment"]["features"] == [
        FEATURE_NAMES[index]
        for index in SEASON_FEATURE_INDICES + HISTORY_FEATURE_INDICES
    ]
    assert set(ablation["with_environment"]["features"]) - set(
        ablation["without_environment"]["features"]
    ) == {"log_rain_30d", "temperature_30d"}
    assert ablation["removed_features"] == ["log_rain_30d", "temperature_30d"]
    assert ablation["season_only_reference"]["features"] == [
        FEATURE_NAMES[index] for index in SEASON_FEATURE_INDICES
    ]

    with_environment = float(ablation["with_environment"]["brier"])
    without_environment = float(ablation["without_environment"]["brier"])
    assert ablation["delta_brier"] == round(with_environment - without_environment, 8)
    assert ablation["environment_reduces_brier"] is (
        with_environment < without_environment
    )

    # Every fold carries its own matched season-only score, so the pooled verdict
    # cannot be an artefact of one origin.
    for fold in evaluation["folds"]:
        assert "season_only_ridge_brier" in fold
        assert 0.0 < float(fold["season_only_ridge_brier"]) < 1.0

    outlook = public_outlook_map(horizon_month=1)
    assert outlook["environment_block_ablation"] == ablation


def test_negative_skill_against_the_null_competitor_reaches_the_served_payload(
    tmp_path,
) -> None:
    """The artefact must carry the verdict and the served payload must repeat it."""

    evaluation = public_outlook_evaluation()
    assert isinstance(evaluation["beats_unconditional_climatology"], bool)
    beats = evaluation["beats_unconditional_climatology"]

    outlook = public_outlook_map(horizon_month=1)
    assert outlook["beats_unconditional_climatology"] is beats
    assert outlook["brier_skill_score_vs_unconditional"] == (
        evaluation["brier_skill_score_vs_unconditional"]
    )

    client = TestClient(create_app(f"sqlite:///{tmp_path / 'skill.sqlite3'}"))
    response = client.get("/api/v1/outlook/public/map?disease=malaria&horizon_month=1")
    payload = response.json()
    codes = {item["code"] for item in payload["warnings"]}
    statement = payload["data"]["model_skill_statement"]
    if beats:
        assert "MODEL_HAS_SKILL_OVER_UNCONDITIONAL_CLIMATOLOGY" in codes
        assert statement.startswith("The selected model beat")
    else:
        # A reader of the served map must be told, in the payload itself, that
        # nothing in the ladder beat a constant. Burying it in the artefact only
        # would leave the map looking like a model result.
        assert "NO_MODEL_BEAT_UNCONDITIONAL_CLIMATOLOGY" in codes
        assert statement.startswith("No model in the evaluated ladder beat a constant")
        assert payload["data"]["brier_skill_score_vs_unconditional"] <= 0.0


def test_served_probability_carries_its_own_source_and_labels_follow_it() -> None:
    """Flags are derived from what was served, not from the selection label."""

    outlook = public_outlook_map(horizon_month=2)
    sources = {row["served_probability_source"] for row in outlook["records"]}
    assert len(sources) == 1
    served_source = sources.pop()
    assert served_source in SERVED_SOURCE_MODEL
    assert outlook["served_probability_source"] == served_source
    assert outlook["selected_model"] == SERVED_SOURCE_MODEL[served_source]
    assert outlook["environment_in_served_vector"] is (
        served_source == "ridge_logistic_environment_season_and_disease_history"
    )
    # Being in the vector is not the same as having earned a place there.
    assert outlook["environment_promoted"] is (
        outlook["environment_in_served_vector"]
        and bool(outlook["environment_block_ablation"]["environment_reduces_brier"])
    )
    assert outlook["served_model_matches_artefact_selection"] is (
        outlook["selected_model"] == outlook["artefact_selected_by_brier"]
    )
    for row in outlook["records"]:
        assert (
            row["environment_used_in_probability"]
            is outlook["environment_in_served_vector"]
        )


def test_environment_scenario_interval_is_null_when_it_was_never_propagated() -> None:
    outlook = public_outlook_map(horizon_month=1)
    for row in outlook["records"]:
        lower = row["probability_lower_environment_scenario"]
        upper = row["probability_upper_environment_scenario"]
        if row["environment_used_in_probability"]:
            assert row["probability_interval_state"].startswith("propagated_from_ensemble")
            assert lower is not None and upper is not None
            assert lower <= row["research_indicator_probability"] <= upper
        else:
            # A degenerate low == central == high implied a propagated interval
            # that does not exist. Absence must be stated, not imitated.
            assert lower is None and upper is None
            assert row["probability_interval_state"] == (
                "not_propagated_environment_is_not_in_the_probability"
            )


def test_target_month_is_day_weighted_so_a_refresh_date_cannot_move_the_number() -> None:
    outlook = public_outlook_map(horizon_month=1)
    for row in outlook["records"]:
        weights = row["target_month_weights"]
        start = date.fromisoformat(row["target_start"])
        end = date.fromisoformat(row["target_end"])
        days = (end - start).days + 1
        counted: dict[str, int] = {}
        for offset in range(days):
            month = str((start + timedelta(days=offset)).month)
            counted[month] = counted.get(month, 0) + 1
        assert set(weights) == set(counted)
        assert sum(weights.values()) == 1.0
        for month, count in counted.items():
            assert weights[month] == round(count / days, 6)

    # The midpoint rule stepped between calendar-month rates, so refreshing the
    # seasonal file one day later could move the served rate by tens of percent.
    # The day-weighted blend has to move continuously instead.
    baseline = public_outlook_evaluation()["serving_baseline"]
    blended = []
    for offset in range(45):
        window_start = date(2026, 7, 1) + timedelta(days=offset)
        window_end = window_start + timedelta(days=29)
        blended.append(_blended_month_rate(baseline, _month_weights(window_start, window_end)))
    largest_one_day_step = max(
        abs(later - earlier) / earlier
        for earlier, later in zip(blended, blended[1:], strict=False)
    )
    assert largest_one_day_step < 0.05


def test_priority_score_publishes_its_rank_correlation_with_official_burden() -> None:
    outlook = public_outlook_map(horizon_month=1)
    independence = outlook["priority_independence"]
    assert independence["method"] == "spearman_rank_correlation_computed_at_request_time"
    assert independence["districts_compared"] == 30
    coefficient = independence["coefficient"]
    # The number itself is the finding: the priority rank is close to a monotone
    # re-expression of the observed burden map, and a reader must see how close.
    assert coefficient is not None
    assert -1.0 <= coefficient <= 1.0
    assert coefficient > 0.9


def test_outlook_envelope_dates_itself_to_the_training_panel_end(tmp_path) -> None:
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'vintage.sqlite3'}"))
    payload = client.get("/api/v1/outlook/public/map?horizon_month=1").json()

    panel_end = max(row.period_start for row in load_hmis_rows())
    vintage = payload["context"]["data_vintage"]
    assert vintage["state"] == "value"
    assert vintage["value"]["vintage_id"] == f"hmis_monthly_panel_end_{panel_end}"

    declared = payload["data"]["training_data_vintage"]
    assert declared["hmis_training_series_end"] == panel_end
    assert declared["last_evaluated_target_year"] == 2019
    assert declared["reason_code"] == "TRAINING_SERIES_ENDS_BEFORE_TARGET_WINDOW"
    assert declared["served_target_window_start"] > panel_end
    assert "TRAINING_SERIES_ENDS_BEFORE_TARGET_WINDOW" in {
        item["code"] for item in payload["warnings"]
    }


def test_a_metric_the_source_never_printed_is_explained_not_left_blank(tmp_path) -> None:
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'blank.sqlite3'}"))

    for year in (2017, 2018):
        payload = client.get(
            f"/api/v1/public-health/malaria/map?year={year}&metric=total_cases"
        ).json()
        records = payload["data"]["records"]
        assert len(records) == 30
        assert all(row["observation_state"] == "not_reported_in_table" for row in records)
        warning = next(
            item
            for item in payload["warnings"]
            if item["code"] == "METRIC_NOT_PRINTED_IN_SOURCE_YEAR"
        )
        assert str(year) in warning["message"]
        assert "total_cases" in warning["message"]
        assert payload["context"]["coverage_state"] == "unavailable"

    # A year that does print the metric must not carry the explanation.
    printed = client.get("/api/v1/public-health/malaria/map?year=2019&metric=total_cases").json()
    assert "METRIC_NOT_PRINTED_IN_SOURCE_YEAR" not in {
        item["code"] for item in printed["warnings"]
    }
    assert printed["context"]["coverage_state"] == "observed_for_registered_sources"


def test_readiness_says_why_the_map_stops_at_the_district(tmp_path) -> None:
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'readiness.sqlite3'}"))
    capabilities = {
        item["capability"]: item
        for item in client.get("/api/v1/readiness").json()["data"]["capabilities"]
    }
    detail = capabilities["tahasil_health_map"]["detail"]
    assert "317 revenue tahasils" in detail
    assert "314 community development" in detail
    assert "CHC/PHC catchments" in detail
    assert "No authorised sub-district surveillance series is available" in detail

