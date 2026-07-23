from packages.forecasting import build_synthetic_report


def test_synthetic_forecast_is_deterministic_and_watermarked() -> None:
    first = build_synthetic_report()
    second = build_synthetic_report()
    assert first == second
    assert first["is_synthetic"] is True
    assert first["watermark"] == "SIMULATION_ONLY_NOT_ODISHA_RISK"
    assert first["real_odisha_prediction_available"] is False
    assert len(first["latest_simulation_map"]) == 30
    assert all(
        row["watermark"] == "SIMULATION_ONLY_NOT_ODISHA_RISK"
        for row in first["latest_simulation_map"]
    )


def test_rolling_origins_are_forward_only() -> None:
    report = build_synthetic_report()
    origins = report["rolling_origins"]
    assert [row["origin_week_index"] for row in origins] == [104, 130, 143]
    assert all(row["train_rows"] > row["test_rows"] > 0 for row in origins)
    assert 0 <= report["pooled"]["model_brier"] <= 1
    assert 0 <= report["pooled"]["seasonal_baseline_brier"] <= 1


def test_three_month_software_path_is_synthetic_and_issue_time_bounded() -> None:
    report = build_synthetic_report(horizon_weeks=12)
    assert report["horizon_weeks"] == 12
    assert report["target"] == "synthetic_12_week_ahead_binary_event"
    assert report["real_odisha_prediction_available"] is False
    assert len(report["latest_simulation_map"]) == 30
    assert all(
        row["target_date"] > row["issue_date"] for row in report["latest_simulation_map"]
    )
