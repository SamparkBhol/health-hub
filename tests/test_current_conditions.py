"""Contract tests for the present-day environmental conditions layer.

The layer's whole reason to exist is that it can speak about *now* while the
reported-outbreak model cannot.  These tests pin the two things that make that
publishable: the near-real-time features are computed the same way the model was
trained, and nothing anywhere in the payload is a case number.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from packages.forecasting.climate import (
    EXTENDED_FEATURE_NAMES,
    FEATURE_NAMES,
    MAXIMUM_TRAILING_WEEKS,
    build_feature_index,
    load_weekly_panel,
    weekly_from_receipt,
)
from packages.forecasting.current_conditions import (
    ALL_FEATURE_NAMES,
    MINIMUM_DAY_COVERAGE,
    NOT_A_FORECAST_WARNING,
    CurrentConditionsUnavailable,
    DailySeries,
    anchor_week,
    build_layer,
    build_recent_features,
    daily_series_from_receipt,
    load_layer,
    map_payload,
)
from packages.forecasting.suitability import (
    BANDS,
    MINIMUM_REFERENCE_SAMPLES,
    QUANTITY,
    SuitabilityArtefactInvalid,
    SuitabilityModel,
    SuitabilityScore,
    band_for,
    load_model,
)
from pipelines.environmental.current import (
    DEFAULT_LOOKBACK_DAYS,
    PROVIDER_LATENCY_DAYS,
    default_window,
    load_recent_receipt,
    read_manifest,
    recent_data_edge,
)

ODISHA_DISTRICT_COUNT = 30
FORBIDDEN_CASE_KEYS = {
    "cases",
    "case_count",
    "cases_reported",
    "incidence",
    "expected_cases",
    "predicted_cases",
    "patients",
    "deaths",
}


def _synthetic_series(district_id: str, end: date, *, drop: set[date] | None = None) -> DailySeries:
    """A dense fabricated daily record, used only to exercise window arithmetic."""

    missing = drop or set()
    rain: dict[date, float] = {}
    t2m: dict[date, float] = {}
    tmax: dict[date, float] = {}
    tmin: dict[date, float] = {}
    rh: dict[date, float] = {}
    for offset in range(200):
        day = end - timedelta(days=offset)
        if day in missing:
            continue
        rain[day] = 2.0 + (offset % 5)
        t2m[day] = 27.0
        tmax[day] = 32.0
        tmin[day] = 24.0
        rh[day] = 85.0
    return DailySeries(district_id, rain, t2m, tmax, tmin, rh)


def test_default_window_respects_provider_latency() -> None:
    today = date(2026, 7, 21)
    start, end = default_window(today)
    assert end == today - timedelta(days=PROVIDER_LATENCY_DAYS)
    assert (end - start).days == DEFAULT_LOOKBACK_DAYS - 1


def test_anchor_week_is_the_last_fully_observed_iso_week() -> None:
    series = _synthetic_series("OD-DIST-puri", date(2026, 7, 19))
    assert anchor_week(series) == date(2026, 7, 13)
    assert anchor_week(DailySeries("x", {}, {}, {}, {}, {})) is None


def test_anchor_week_steps_back_over_a_broken_week() -> None:
    series = _synthetic_series("OD-DIST-puri", date(2026, 7, 19), drop={date(2026, 7, 15)})
    assert anchor_week(series) == date(2026, 7, 6)


def test_anchor_week_requires_every_parameter_not_only_rain() -> None:
    series = _synthetic_series("OD-DIST-puri", date(2026, 7, 19))
    t2m = dict(series.t2m_c)
    t2m.pop(date(2026, 7, 15))
    incomplete = DailySeries(
        series.district_id,
        series.rain_mm,
        t2m,
        series.tmax_c,
        series.tmin_c,
        series.rh_pct,
    )
    assert anchor_week(incomplete) == date(2026, 7, 6)


def test_recent_features_match_the_model_feature_width() -> None:
    panel = load_weekly_panel()
    climatology = build_feature_index(panel)
    series = _synthetic_series("OD-DIST-puri", date(2022, 12, 25))
    features = build_recent_features(series, climatology["OD-DIST-puri"])
    assert features is not None
    assert features.status == "observed"
    assert len(features.values) == len(ALL_FEATURE_NAMES)
    assert ALL_FEATURE_NAMES[: len(FEATURE_NAMES)] == FEATURE_NAMES
    assert features.coverage == pytest.approx(1.0)
    assert features.expected_days == MAXIMUM_TRAILING_WEEKS * 7


def test_recent_features_refuse_a_gappy_window() -> None:
    panel = load_weekly_panel()
    climatology = build_feature_index(panel)
    end = date(2022, 12, 25)
    # Punch out enough days, all in earlier weeks, to break the coverage floor
    # while leaving the anchor week itself intact.
    drop = {end - timedelta(days=offset) for offset in range(20, 40)}
    series = _synthetic_series("OD-DIST-puri", end, drop=drop)
    features = build_recent_features(series, climatology["OD-DIST-puri"])
    assert features is not None
    assert features.status == "insufficient_evidence"
    assert features.reason_code is not None
    assert features.values == ()
    assert features.coverage < MINIMUM_DAY_COVERAGE


def test_recent_features_refuse_sparse_non_rain_parameter() -> None:
    panel = load_weekly_panel()
    climatology = build_feature_index(panel)
    end = date(2022, 12, 25)
    series = _synthetic_series("OD-DIST-puri", end)
    # Preserve the complete anchor week but remove older temperature days from
    # the required four-week window. Rain remains 100% complete.
    t2m = {
        day: value
        for day, value in series.t2m_c.items()
        if not (end - timedelta(days=20) <= day <= end - timedelta(days=8))
    }
    sparse = DailySeries(
        series.district_id,
        series.rain_mm,
        t2m,
        series.tmax_c,
        series.tmin_c,
        series.rh_pct,
    )
    features = build_recent_features(sparse, climatology["OD-DIST-puri"])
    assert features is not None
    assert features.status == "insufficient_evidence"
    assert features.reason_code == "RECENT_REQUIRED_PARAMETER_BELOW_MINIMUM_DAY_COVERAGE"
    assert features.parameter_day_coverage["rain_mm"] == pytest.approx(1.0)
    assert features.parameter_day_coverage["t2m_c"] < MINIMUM_DAY_COVERAGE


def test_daily_window_reproduces_the_weekly_training_features() -> None:
    """The near-real-time path must be arithmetic-identical when nothing is missing.

    This is the load-bearing claim of the whole current-conditions layer: the
    features scored against the historical fit are the *same* features the fit
    was trained on.  It is checked against the gap-free historical cache so the
    assertion never softens into a skip because a provider happened to drop a
    day this week.
    """

    from pipelines.environmental.historical import load_receipt
    from pipelines.environmental.historical import read_manifest as read_historical

    panel = load_weekly_panel()
    climatology = build_feature_index(panel)
    historical = read_historical()
    assert historical, "the historical NASA POWER cache is required"
    checked = 0
    for district_id in sorted(historical)[:3]:
        receipt = load_receipt(historical[district_id])
        series = daily_series_from_receipt(district_id, receipt)
        features = build_recent_features(series, climatology[district_id])
        assert features is not None, district_id
        assert features.status == "observed", features.reason_code
        assert features.coverage == pytest.approx(1.0)
        weekly_path = climatology[district_id].extended_features(features.issue_week)
        assert weekly_path is not None, district_id
        for name, daily_value, weekly_value in zip(
            ALL_FEATURE_NAMES, features.values, weekly_path, strict=True
        ):
            assert daily_value == pytest.approx(weekly_value, rel=1e-9, abs=1e-9), (
                f"{district_id}/{name}"
            )
        checked += 1
    assert checked == 3


def test_recent_cache_path_produces_scoreable_features_today() -> None:
    """The live near-real-time cache must actually yield features, not just parse."""

    manifest = read_manifest()
    if not manifest:  # pragma: no cover - cache is built by the collector
        pytest.skip("no recent cache")
    climatology = build_feature_index(load_weekly_panel())
    scored = 0
    for district_id, vintage in sorted(manifest.items()):
        receipt = load_recent_receipt(vintage)
        series = daily_series_from_receipt(district_id, receipt)
        weeks = weekly_from_receipt(receipt)
        assert weeks, district_id
        features = build_recent_features(series, climatology[district_id])
        assert features is not None, district_id
        if features.status == "observed":
            assert len(features.values) == len(ALL_FEATURE_NAMES)
            assert features.coverage >= MINIMUM_DAY_COVERAGE
            scored += 1
        else:
            assert features.reason_code
    assert scored == ODISHA_DISTRICT_COUNT


def test_band_boundaries_are_ordered() -> None:
    assert band_for(10.0) == "below_typical"
    assert band_for(60.0) == "typical"
    assert band_for(80.0) == "elevated"
    assert band_for(99.0) == "much_above_typical"
    assert band_for(100.0) == "much_above_typical"
    assert [label for _ceiling, label in BANDS] == [
        "below_typical",
        "typical",
        "elevated",
        "much_above_typical",
    ]


def test_suitability_artefact_must_declare_it_is_not_incidence() -> None:
    with pytest.raises(SuitabilityArtefactInvalid, match="is_synthetic"):
        SuitabilityModel({"schema_version": "1.0.0", "is_synthetic": True})
    with pytest.raises(SuitabilityArtefactInvalid, match="incidence"):
        SuitabilityModel(
            {
                "schema_version": "1.0.0",
                "is_synthetic": False,
                "quantity": {"is_incidence": True, "is_case_count": False},
            }
        )


def test_fitted_suitability_model_scores_and_refuses_correctly() -> None:
    model = load_model()
    assert model.payload["quantity"]["kind"] == QUANTITY
    assert model.payload["quantity"]["is_case_count"] is False
    assert model.payload["quantity"]["is_outbreak_probability"] is False
    assert model.feature_names == list(FEATURE_NAMES)
    assert len(EXTENDED_FEATURE_NAMES) == 5

    zeros = [0.0] * len(FEATURE_NAMES)
    scored = model.score("OD-DIST-puri", 28, zeros)
    assert scored.status == "scored"
    assert 0.0 <= float(scored.percentile) <= 100.0
    assert scored.reference_samples >= MINIMUM_REFERENCE_SAMPLES
    assert scored.drivers

    unknown = model.score("OD-DIST-not-a-district", 28, zeros)
    assert unknown.status == "insufficient_reference"
    assert unknown.percentile is None
    assert unknown.reason_code == "NO_SEASONAL_REFERENCE_DISTRIBUTION_FOR_DISTRICT_WEEK"


def test_percentile_scale_is_uniform_on_its_own_reference_period() -> None:
    """A percentile that is not uniform on its reference period is not a percentile.

    Scoring every historical modelling row back through the published quantile
    grid must land roughly one tenth of them in each decile.  If this drifts, a
    reading of "92nd percentile" no longer means what the layer claims it means.
    """

    from packages.forecasting.backtest import PANEL_END, PANEL_START
    from packages.forecasting.panel import build_examples, panel_weeks
    from packages.forecasting.target import build_target_panel

    model = load_model()
    climate = build_feature_index(load_weekly_panel())
    rows = build_examples(
        target_panel=build_target_panel("any_reported_outbreak"),
        climate=climate,
        horizon_weeks=1,
        weeks=panel_weeks(PANEL_START, PANEL_END),
    )
    width = len(FEATURE_NAMES)
    deciles = [0] * 10
    scored = 0
    for row in rows:
        result = model.score(
            row.district_id, row.issue_week.isocalendar().week, row.features[:width]
        )
        if result.percentile is None:
            continue
        deciles[min(int(result.percentile // 10), 9)] += 1
        scored += 1
    assert scored > 10_000
    for index, count in enumerate(deciles):
        share = count / scored
        assert 0.085 <= share <= 0.115, f"decile {index} holds {share:.3%} of rows"


def test_suitability_rejects_a_wrong_width_feature_vector() -> None:
    model = load_model()
    with pytest.raises(ValueError, match="environmental features"):
        model.linear_predictor([0.0, 0.0])


def test_recent_cache_covers_every_district() -> None:
    manifest = read_manifest()
    if not manifest:  # pragma: no cover
        pytest.skip("no recent cache")
    assert len(manifest) == ODISHA_DISTRICT_COUNT
    edge = recent_data_edge()
    assert edge is not None
    for vintage in manifest.values():
        assert vintage.observed_days > 0
        assert vintage.last_observed_day is not None


def _walk(node, keys: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert key.lower() not in FORBIDDEN_CASE_KEYS, key
            _walk(value, keys)
    elif isinstance(node, list):
        for item in node:
            _walk(item, keys)


def test_published_layer_never_carries_a_case_count() -> None:
    try:
        payload = load_layer()
    except CurrentConditionsUnavailable:  # pragma: no cover
        pytest.skip("no current-conditions layer built")
    assert payload["is_synthetic"] is False
    assert payload["quantity"]["is_case_count"] is False
    assert payload["quantity"]["is_incidence"] is False
    assert payload["quantity"]["is_outbreak_probability"] is False
    assert payload["quantity"]["is_forecast"] is False
    assert NOT_A_FORECAST_WARNING in payload["warnings"]
    _walk(payload, FORBIDDEN_CASE_KEYS)


def test_published_layer_covers_every_district_and_flags_each_row() -> None:
    try:
        payload = load_layer()
    except CurrentConditionsUnavailable:  # pragma: no cover
        pytest.skip("no current-conditions layer built")
    districts = payload["districts"]
    assert len(districts) == ODISHA_DISTRICT_COUNT
    assert len({row["district_id"] for row in districts}) == ODISHA_DISTRICT_COUNT
    shared = payload["sources"]["climate"]["districts_sharing_a_grid_cell"]
    for row in districts:
        assert row["is_synthetic"] is False
        assert (row["district_id"] in shared) == bool(row["shares_climate_grid_cell_with"])
        assert row["status"] in {"observed", "partial", "insufficient_evidence"}
        if row["status"] == "insufficient_evidence":
            assert row["reason_code"], row["district_id"]
        else:
            percentile = row["suitability"]["suitability_percentile"]
            assert percentile is None or 0.0 <= percentile <= 100.0


def test_map_payload_is_render_ready_and_still_honest() -> None:
    try:
        payload = map_payload()
    except CurrentConditionsUnavailable:  # pragma: no cover
        pytest.skip("no current-conditions layer built")
    assert payload["is_synthetic"] is False
    assert len(payload["districts"]) == ODISHA_DISTRICT_COUNT
    _walk(payload, FORBIDDEN_CASE_KEYS)
    for row in payload["districts"]:
        assert row["is_synthetic"] is False
        assert set(row) >= {
            "district_id",
            "canonical_name",
            "status",
            "suitability_percentile",
            "band",
            "imd_peak_warning_next_5_days",
        }


def test_zero_percentile_is_still_counted_and_ranked() -> None:
    if not read_manifest():  # pragma: no cover - cache is built by collector
        pytest.skip("no recent cache")

    class ZeroPercentileModel:
        generated_at = "2026-07-21T00:00:00Z"
        payload = {
            "model_version": "test-zero-percentile",
            "fitted_against": "test",
            "warnings": [],
        }

        def score(self, district_id, iso_week, values):  # noqa: ANN001
            return SuitabilityScore(
                district_id=district_id,
                iso_week=iso_week,
                linear_predictor=-99.0,
                percentile=0.0,
                band="below_typical",
                reference_samples=100,
                drivers=(),
                status="scored",
            )

    payload = build_layer(suitability=ZeroPercentileModel())  # type: ignore[arg-type]
    assert payload["coverage"]["scored"] == ODISHA_DISTRICT_COUNT
    assert len(payload["ranking_most_unusual_conditions"]) == 10
    assert all(
        row["suitability_percentile"] == 0.0 for row in payload["ranking_most_unusual_conditions"]
    )


def test_layer_reader_rejects_a_synthetic_payload(tmp_path: Path) -> None:
    path = tmp_path / "layer.json"
    path.write_text(json.dumps({"is_synthetic": True}), encoding="utf-8")
    with pytest.raises(CurrentConditionsUnavailable, match="is_synthetic"):
        load_layer(path)


def test_layer_reader_rejects_a_case_count_payload(tmp_path: Path) -> None:
    path = tmp_path / "layer.json"
    path.write_text(
        json.dumps({"is_synthetic": False, "quantity": {"is_case_count": True}}),
        encoding="utf-8",
    )
    with pytest.raises(CurrentConditionsUnavailable, match="incidence nor a case count"):
        load_layer(path)


def test_layer_reader_rejects_an_outbreak_probability_payload(tmp_path: Path) -> None:
    path = tmp_path / "layer.json"
    path.write_text(
        json.dumps(
            {
                "is_synthetic": False,
                "quantity": {
                    "is_case_count": False,
                    "is_incidence": False,
                    "is_outbreak_probability": True,
                    "is_forecast": False,
                },
                "districts": [{}] * ODISHA_DISTRICT_COUNT,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CurrentConditionsUnavailable, match="outbreak probability"):
        load_layer(path)


def test_layer_reader_reports_a_missing_artefact(tmp_path: Path) -> None:
    with pytest.raises(CurrentConditionsUnavailable, match="no current-conditions layer"):
        load_layer(tmp_path / "absent.json")


def test_environment_block_ablation_artefact_is_honest_about_its_null_result() -> None:
    """The ablation must publish its own negative finding, not hide it."""

    from packages.forecasting.ablation import VARIANT_ORDER, load_ablation

    try:
        payload = load_ablation()
    except FileNotFoundError:  # pragma: no cover - artefact is generated by the CLI
        pytest.skip("no ablation artefact built")
    assert payload["is_synthetic"] is False
    assert payload["design"]["ridge_penalty"] == pytest.approx(8.0)
    assert "identical" in payload["design"]["rows"]
    assert payload["conclusion"]
    assert payload["warnings"]
    evaluated = [cell for cell in payload["cells"] if cell["status"] == "evaluated"]
    assert len(evaluated) >= 8
    for cell in evaluated:
        assert set(cell["variants"]) == set(VARIANT_ORDER)
        reference = cell["variants"]["no_environment"]["brier"]
        for name in VARIANT_ORDER:
            variant = cell["variants"][name]
            assert 0.0 <= variant["brier"] <= 1.0
            assert variant["auc"] is None or 0.0 <= variant["auc"] <= 1.0
        # The whole point of the artefact: the environmental block moves the
        # score by a negligible amount. If this ever stops being true the
        # conclusion text is stale and must be rewritten.
        for name in ("environment_v1_0_0", "environment_v1_1_0"):
            change = abs(cell["variants"][name]["brier"] - reference)
            assert change < 1e-3, (name, cell["horizon_weeks"], change)


def test_published_model_features_match_the_shipped_artefact() -> None:
    """The reverted feature block must still describe the artefact that ships."""

    import json as _json

    from packages.forecasting.backtest import ARTEFACT_PATH, FEATURE_BLOCKS

    payload = _json.loads(ARTEFACT_PATH.read_text(encoding="utf-8"))
    assert payload["features"]["names"][: len(FEATURE_NAMES)] == list(FEATURE_NAMES)
    assert payload["features"]["blocks"]["environment"] == list(FEATURE_BLOCKS["environment"])
    assert FEATURE_BLOCKS["environment"] == (0, len(FEATURE_NAMES))
