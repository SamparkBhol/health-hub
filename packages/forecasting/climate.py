"""Weekly environmental features from the cached NASA POWER daily district panel.

Every feature is defined so that it is computable from information available at
its ISO issue week: trailing windows never look forward, and the anomaly
climatology is an *expanding* window that only uses strictly earlier calendar
years.  Nothing here is disease data.

NASA POWER is a coarse global reanalysis sampled at one representative interior
point per district.  It is environmental context for a district, not a
district-average exposure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from pipelines.environmental.historical import CachedVintage, load_receipt, read_manifest

from .target import week_start

PARAMETERS = ("PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M")
MIN_PRIOR_YEARS = 3
WET_DAY_MM = 1.0
HUMID_DAY_PCT = 80.0
#: Trailing window, in weeks, that the longest environmental feature needs.
MAXIMUM_TRAILING_WEEKS = 8

#: Ten columns, and deliberately still ten.  An expansion to fifteen - an
#: 8-week rainfall anomaly, the longest dry run, a humid-day count, mean daily
#: maximum temperature and an explicit rain x temperature interaction - was
#: built, fitted and scored against this block on identical rows, origins and
#: baselines by :mod:`packages.forecasting.ablation`.  It did not improve skill
#: at any published horizon and was measurably worse at four weeks, so it was
#: not shipped.  With 294 positive district-weeks in 18450 rows there is no
#: budget for columns that do not earn their place.
FEATURE_NAMES: tuple[str, ...] = (
    "rain_1w_mm",
    "rain_2w_mm",
    "rain_4w_mm",
    "rain_8w_mm",
    "rain_4w_anomaly_sd",
    "t2m_4w_c",
    "t2m_4w_anomaly_sd",
    "rh_4w_pct",
    "rh_4w_anomaly_sd",
    "dtr_4w_c",
)

FEATURE_NOTES: dict[str, str] = {
    "rain_1w_mm": "corrected precipitation total over the issue ISO week",
    "rain_2w_mm": "2-week trailing precipitation accumulation ending at the issue week",
    "rain_4w_mm": "4-week trailing precipitation accumulation ending at the issue week",
    "rain_8w_mm": "8-week trailing precipitation accumulation ending at the issue week",
    "rain_4w_anomaly_sd": (
        "4-week accumulation minus the expanding same-ISO-week mean from strictly "
        "earlier years, scaled by the expanding standard deviation"
    ),
    "t2m_4w_c": "4-week trailing mean 2 m air temperature",
    "t2m_4w_anomaly_sd": "expanding same-ISO-week temperature anomaly in standard deviations",
    "rh_4w_pct": "4-week trailing mean 2 m relative humidity",
    "rh_4w_anomaly_sd": "expanding same-ISO-week humidity anomaly in standard deviations",
    "dtr_4w_c": "4-week trailing mean diurnal temperature range (T2M_MAX - T2M_MIN)",
}

#: Candidate columns kept available for the ablation harness and for describing
#: current conditions, but NOT part of the published model's feature vector.
EXTENDED_FEATURE_NAMES: tuple[str, ...] = (
    "rain_8w_anomaly_sd",
    "longest_dry_run_4w_days",
    "humid_days_4w",
    "tmax_4w_c",
    "rain_4w_x_t2m_4w",
)
EXTENDED_FEATURE_NOTES: dict[str, str] = {
    "rain_8w_anomaly_sd": (
        "8-week accumulation anomaly against the expanding same-ISO-week "
        "climatology from strictly earlier years"
    ),
    "longest_dry_run_4w_days": (
        f"longest run of consecutive days below {WET_DAY_MM:g} mm in the trailing 4 weeks"
    ),
    "humid_days_4w": (
        f"days with mean relative humidity at or above {HUMID_DAY_PCT:g} percent "
        "in the trailing 4 weeks"
    ),
    "tmax_4w_c": "4-week trailing mean daily maximum temperature",
    "rain_4w_x_t2m_4w": (
        "product of the 4-week rain accumulation and the 4-week mean temperature "
        "centred at a fixed reference; an explicit warm-and-wet interaction"
    ),
}

#: The interaction term is formed around a fixed reference value rather than the
#: sample mean, so a refit on a different training window cannot silently change
#: what the interaction means.
REFERENCE_T2M_C = 27.0


@dataclass(frozen=True, slots=True)
class WeeklyClimate:
    week: date
    days: int
    rain_mm: float
    t2m_c: float
    tmax_c: float
    tmin_c: float
    rh_pct: float
    #: Days at or above WET_DAY_MM. Not a model feature: the wet-day count was
    #: fitted and rejected by the ablation. Kept because the dry-run feature and
    #: the current-conditions layer both reason about the same threshold.
    wet_days: int = 0
    humid_days: int = 0
    #: Daily precipitation in calendar order, kept so dry-spell runs can be
    #: measured across week boundaries rather than only inside a week.
    daily_rain_mm: tuple[float, ...] = ()

    @property
    def complete(self) -> bool:
        return self.days == 7


def weekly_from_receipt(receipt) -> tuple[WeeklyClimate, ...]:
    """Aggregate a validated daily POWER receipt to complete ISO weeks."""

    buckets: dict[date, dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in PARAMETERS}
    )
    daily: dict[date, dict[str, dict[date, float]]] = defaultdict(
        lambda: {name: {} for name in PARAMETERS}
    )
    for value in receipt.values:
        if value.is_fill_value or value.value is None:
            continue
        week = week_start(value.day)
        buckets[week][value.parameter].append(float(value.value))
        daily[week][value.parameter][value.day] = float(value.value)
    weeks: list[WeeklyClimate] = []
    for week, series in sorted(buckets.items()):
        counts = {name: len(values) for name, values in series.items()}
        days = min(counts.values())
        if days == 0 or len(set(counts.values())) != 1:
            continue
        rain_by_day = daily[week]["PRECTOTCORR"]
        humidity_by_day = daily[week]["RH2M"]
        ordered_rain = tuple(rain_by_day[day] for day in sorted(rain_by_day))
        weeks.append(
            WeeklyClimate(
                week=week,
                days=days,
                rain_mm=sum(series["PRECTOTCORR"]),
                t2m_c=sum(series["T2M"]) / days,
                tmax_c=sum(series["T2M_MAX"]) / days,
                tmin_c=sum(series["T2M_MIN"]) / days,
                rh_pct=sum(series["RH2M"]) / days,
                wet_days=sum(1 for value in ordered_rain if value >= WET_DAY_MM),
                humid_days=sum(
                    1 for value in humidity_by_day.values() if value >= HUMID_DAY_PCT
                ),
                daily_rain_mm=ordered_rain,
            )
        )
    return tuple(weeks)


def load_weekly_panel(
    root: Path | None = None,
    manifest: dict[str, CachedVintage] | None = None,
) -> dict[str, tuple[WeeklyClimate, ...]]:
    vintages = manifest if manifest is not None else read_manifest(root)
    if not vintages:
        raise FileNotFoundError(
            "no cached NASA POWER district vintages; run "
            "`python scripts/collect_environment.py --mode historical-panel "
            "--start 2008-01-01 --end 2022-12-31`"
        )
    return {
        district_id: weekly_from_receipt(load_receipt(vintage, root))
        for district_id, vintage in sorted(vintages.items())
    }


class DistrictClimateFeatures:
    """Issue-time environmental features for one district."""

    def __init__(self, weeks: tuple[WeeklyClimate, ...]) -> None:
        self.weeks = [item for item in weeks if item.complete]
        self.index = {item.week: position for position, item in enumerate(self.weeks)}
        self._rain = [item.rain_mm for item in self.weeks]
        self._t2m = [item.t2m_c for item in self.weeks]
        self._tmax = [item.tmax_c for item in self.weeks]
        self._rh = [item.rh_pct for item in self.weeks]
        self._dtr = [item.tmax_c - item.tmin_c for item in self.weeks]
        self._humid = [item.humid_days for item in self.weeks]
        self._daily_rain = [item.daily_rain_mm for item in self.weeks]
        self._cache: dict[date, tuple[float, ...] | None] = {}
        self._history: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
        self._build_history()

    def _window_mean(self, series: list[float], position: int, span: int) -> float | None:
        if position + 1 < span:
            return None
        window = series[position + 1 - span : position + 1]
        return sum(window) / span

    def _build_history(self) -> None:
        """Index every (variable, ISO week number) -> [(iso_year, value)] for anomalies."""

        for position, item in enumerate(self.weeks):
            calendar = item.week.isocalendar()
            rain4 = self._window_mean(self._rain, position, 4)
            rain8 = self._window_mean(self._rain, position, 8)
            t2m4 = self._window_mean(self._t2m, position, 4)
            rh4 = self._window_mean(self._rh, position, 4)
            for name, value in (
                ("rain4", rain4),
                ("rain8", rain8),
                ("t2m4", t2m4),
                ("rh4", rh4),
            ):
                if value is not None:
                    self._history[(name, calendar.week)].append((calendar.year, value))

    def _anomaly(self, name: str, week: date, value: float) -> float | None:
        calendar = week.isocalendar()
        prior = [
            observed
            for year, observed in self._history[(name, calendar.week)]
            if year < calendar.year
        ]
        if len(prior) < MIN_PRIOR_YEARS:
            return None
        mean = sum(prior) / len(prior)
        variance = sum((item - mean) ** 2 for item in prior) / len(prior)
        deviation = math.sqrt(variance)
        if deviation < 1e-6:
            return 0.0
        return (value - mean) / deviation

    def anomaly(self, name: str, week: date, value: float) -> float | None:
        """Standardised anomaly of ``value`` against strictly earlier calendar years.

        Public because the near-real-time path forms the same anomalies from a
        daily window while borrowing this district's historical climatology.
        Valid names are ``rain4``, ``rain8``, ``t2m4`` and ``rh4``.
        """

        return self._anomaly(name, week, value)

    def _longest_dry_run(self, position: int, weeks_back: int) -> int:
        """Longest run of sub-threshold days across the trailing ``weeks_back`` weeks."""

        longest = 0
        running = 0
        for offset in range(weeks_back - 1, -1, -1):
            for value in self._daily_rain[position - offset]:
                if value < WET_DAY_MM:
                    running += 1
                    longest = max(longest, running)
                else:
                    running = 0
        return longest

    def features(self, issue_week: date) -> tuple[float, ...] | None:
        """The published model's feature vector at ``issue_week``, or ``None``."""

        full = self.extended_features(issue_week)
        return None if full is None else full[: len(FEATURE_NAMES)]

    def extended_features(self, issue_week: date) -> tuple[float, ...] | None:
        """:data:`FEATURE_NAMES` followed by :data:`EXTENDED_FEATURE_NAMES`.

        The extended columns are computed but not published: they exist so that
        :mod:`packages.forecasting.ablation` can score them against the shipped
        block on identical rows, and so the current-conditions layer can describe
        conditions with them.  Nothing in the published model reads past
        ``len(FEATURE_NAMES)``.
        """

        if issue_week in self._cache:
            return self._cache[issue_week]
        position = self.index.get(issue_week)
        result: tuple[float, ...] | None = None
        # A trailing window is only formed over contiguous complete weeks: a
        # dropped week must not be silently bridged by its neighbours.
        span = MAXIMUM_TRAILING_WEEKS - 1
        contiguous = (
            position is not None
            and position >= span
            and (issue_week - self.weeks[position - span].week).days == 7 * span
        )
        if position is not None and contiguous:
            rain1 = self._rain[position]
            rain2 = sum(self._rain[position - 1 : position + 1])
            rain4 = sum(self._rain[position - 3 : position + 1])
            rain8 = sum(self._rain[position - 7 : position + 1])
            t2m4 = sum(self._t2m[position - 3 : position + 1]) / 4.0
            tmax4 = sum(self._tmax[position - 3 : position + 1]) / 4.0
            rh4 = sum(self._rh[position - 3 : position + 1]) / 4.0
            dtr4 = sum(self._dtr[position - 3 : position + 1]) / 4.0
            humid4 = float(sum(self._humid[position - 3 : position + 1]))
            dry_run4 = float(self._longest_dry_run(position, 4))
            rain_anom = self._anomaly("rain4", issue_week, rain4 / 4.0)
            rain8_anom = self._anomaly("rain8", issue_week, rain8 / 8.0)
            t2m_anom = self._anomaly("t2m4", issue_week, t2m4)
            rh_anom = self._anomaly("rh4", issue_week, rh4)
            if None not in (rain_anom, rain8_anom, t2m_anom, rh_anom):
                result = (
                    rain1,
                    rain2,
                    rain4,
                    rain8,
                    float(rain_anom),  # type: ignore[arg-type]
                    t2m4,
                    float(t2m_anom),  # type: ignore[arg-type]
                    rh4,
                    float(rh_anom),  # type: ignore[arg-type]
                    dtr4,
                    float(rain8_anom),  # type: ignore[arg-type]
                    dry_run4,
                    humid4,
                    tmax4,
                    rain4 * (t2m4 - REFERENCE_T2M_C),
                )
        self._cache[issue_week] = result
        return result


def build_feature_index(
    panel: dict[str, tuple[WeeklyClimate, ...]],
) -> dict[str, DistrictClimateFeatures]:
    return {
        district_id: DistrictClimateFeatures(weeks) for district_id, weeks in panel.items()
    }


def iso_weeks_between(start: date, end: date) -> list[date]:
    weeks: list[date] = []
    current = week_start(start)
    limit = week_start(end)
    while current <= limit:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks
