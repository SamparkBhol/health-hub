"""Historical experiment: environment + EpiClim row history -> row occurrence.

Every row answers a deliberately narrow question: *for historical ISO week
``t + h``, does the frozen EpiClim file contain a matching district row?* This
does not establish what an analyst actually knew at ``t`` because EpiClim has no
publication timestamp or revision history. The panel is suitable for a bounded
retrospective experiment, not an operational disease forecast.

Two rules limit target leakage within this retrospective convention:

* environmental features use trailing windows ending at ``t`` and anomaly
  climatologies built from strictly earlier calendar years only;
* EpiClim row history is truncated by a declared sensitivity lag. The lag is an
  assumption, not an observed publication delay, and cannot prove issue-time
  availability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from .climate import FEATURE_NAMES as CLIMATE_FEATURE_NAMES
from .climate import DistrictClimateFeatures
from .target import TargetPanel, week_start

DEFAULT_REPORT_LAG_WEEKS = 2
HISTORY_START = date(2009, 1, 1)
MAX_WEEKS_SINCE_EVENT = 260

SEASONAL_FEATURE_NAMES = (
    "target_annual_sin",
    "target_annual_cos",
    "target_semiannual_sin",
    "target_semiannual_cos",
)
HISTORY_FEATURE_NAMES = (
    "district_reports_4w",
    "district_reports_52w",
    "state_reports_4w",
    "district_expanding_report_logit",
    "log1p_weeks_since_district_report",
)
PANEL_FEATURE_NAMES: tuple[str, ...] = (
    CLIMATE_FEATURE_NAMES + SEASONAL_FEATURE_NAMES + HISTORY_FEATURE_NAMES
)


@dataclass(frozen=True, slots=True)
class Example:
    district_id: str
    issue_week: date
    target_week: date
    target_week_of_year: int
    features: tuple[float, ...]
    target: int


def _week_index(weeks: list[date]) -> dict[date, int]:
    return {week: position for position, week in enumerate(weeks)}


def build_examples(
    *,
    target_panel: TargetPanel,
    climate: dict[str, DistrictClimateFeatures],
    horizon_weeks: int,
    weeks: list[date],
    report_lag_weeks: int = DEFAULT_REPORT_LAG_WEEKS,
    history_start: date = HISTORY_START,
    extended: bool = False,
) -> list[Example]:
    """Build every retrospective (district, issue week) row for one horizon.

    ``extended`` swaps the environmental block for the wider candidate vector
    that :mod:`packages.forecasting.ablation` scores. The retained experiment never
    sets it; the extra columns exist only so the ablation can compare feature
    sets on rows that are otherwise byte-identical.
    """

    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1")
    if report_lag_weeks < 0:
        raise ValueError("report_lag_weeks must not be negative")
    positions = _week_index(weeks)
    districts = sorted(climate)
    observed: dict[str, set[int]] = {district: set() for district in districts}
    for district_id, week in target_panel.district_weeks:
        position = positions.get(week)
        if position is not None and district_id in observed:
            observed[district_id].add(position)

    history_position = positions.get(week_start(history_start), 0)

    # Cumulative report counts by week position, per district and state-wide.
    total = len(weeks)
    cumulative: dict[str, list[int]] = {}
    state_counts = [0] * total
    for district_id in districts:
        running = 0
        column = [0] * (total + 1)
        for position in range(total):
            if position in observed[district_id]:
                running += 1
                state_counts[position] += 1
            column[position + 1] = running
        cumulative[district_id] = column
    state_cumulative = [0] * (total + 1)
    running = 0
    for position in range(total):
        running += state_counts[position]
        state_cumulative[position + 1] = running

    last_event: dict[str, list[int | None]] = {}
    for district_id in districts:
        seen: int | None = None
        history: list[int | None] = []
        for position in range(total):
            if position in observed[district_id]:
                seen = position
            history.append(seen)
        last_event[district_id] = history

    def window(column: list[int], low: int, high: int) -> int:
        low = max(low, 0)
        high = max(min(high, total - 1), -1)
        if high < low:
            return 0
        return column[high + 1] - column[low]

    examples: list[Example] = []
    for district_id in districts:
        features_source = climate[district_id]
        district_column = cumulative[district_id]
        district_last = last_event[district_id]
        for position, issue_week in enumerate(weeks):
            target_position = position + horizon_weeks
            if target_position >= total:
                continue
            environment = (
                features_source.extended_features(issue_week)
                if extended
                else features_source.features(issue_week)
            )
            if environment is None:
                continue
            cutoff = position - report_lag_weeks
            if cutoff < history_position + 3:
                continue
            target_week = weeks[target_position]
            calendar = target_week.isocalendar()
            phase = 2.0 * math.pi * (calendar.week - 1) / 52.1775
            reports_4w = window(district_column, cutoff - 3, cutoff)
            reports_52w = window(district_column, cutoff - 51, cutoff)
            state_4w = window(state_cumulative, cutoff - 3, cutoff)
            elapsed = cutoff - history_position + 1
            events_to_date = window(district_column, history_position, cutoff)
            rate = (events_to_date + 0.5) / (elapsed - events_to_date + 0.5)
            previous = district_last[cutoff] if 0 <= cutoff < total else None
            since = (
                MAX_WEEKS_SINCE_EVENT
                if previous is None
                else min(MAX_WEEKS_SINCE_EVENT, cutoff - previous)
            )
            features = (
                *environment,
                math.sin(phase),
                math.cos(phase),
                math.sin(2.0 * phase),
                math.cos(2.0 * phase),
                float(reports_4w),
                float(reports_52w),
                float(state_4w),
                math.log(rate),
                math.log1p(since),
            )
            examples.append(
                Example(
                    district_id=district_id,
                    issue_week=issue_week,
                    target_week=target_week,
                    target_week_of_year=calendar.week,
                    features=features,
                    target=int(target_position in observed[district_id]),
                )
            )
    examples.sort(key=lambda row: (row.issue_week, row.district_id))
    return examples


def panel_weeks(start: date, end: date) -> list[date]:
    """Contiguous ISO week Mondays covering ``start``..``end`` inclusive."""

    weeks: list[date] = []
    current = week_start(start)
    limit = week_start(end)
    while current <= limit:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks
