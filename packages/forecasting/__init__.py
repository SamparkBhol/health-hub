"""Forecasting contracts: a synthetic harness and a historical data experiment.

Two strictly separate paths live here and must never be blended.

``build_synthetic_report`` is the deterministic simulation harness.  It exercises
issue-time feature construction and rolling-origin scoring on invented data and
is watermarked ``SIMULATION_ONLY_NOT_ODISHA_RISK``.  It carries no information
about Odisha.

The historical path (:mod:`packages.forecasting.backtest`,
:mod:`packages.forecasting.service`) fits real NASA POWER climate against the
frozen EpiClim Odisha catalogue. It estimates only whether that incomplete file
contains a matching district-week row. It is explicitly experimental: not
incidence, not a case count, not an official-publication probability and not an
operational disease forecast. Cells that do not beat a seasonal climatology
baseline under the declared rolling-origin convention return
``insufficient_evidence`` instead of a number.

Real-data names resolve lazily so that importing this package for the synthetic
harness does not drag in the climate cache reader, and so that
``python -m packages.forecasting.backtest`` does not double-import itself.
"""

from __future__ import annotations

from typing import Any

from .synthetic import build_synthetic_report

_LAZY: dict[str, str] = {
    "ForecastArtefactInvalid": "service",
    "ForecastArtefactMissing": "service",
    "current_week_refusal": "service",
    "evaluation": "service",
    "load_report": "service",
    "probability_map": "service",
    "summary": "service",
    "run_backtest": "backtest",
    "NOT_INCIDENCE_WARNING": "backtest",
    "TARGET_KIND": "target",
    "TARGET_STATEMENT": "target",
    # Present-day environmental conditions. Separate quantity, separate
    # artefact, and never a probability of disease - see the module docstrings.
    "CurrentConditionsUnavailable": "current_conditions",
    "NOT_A_FORECAST_WARNING": "current_conditions",
    "current_conditions_layer": "current_conditions",
    "current_conditions_map": "current_conditions",
}

__all__ = ["build_synthetic_report", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
