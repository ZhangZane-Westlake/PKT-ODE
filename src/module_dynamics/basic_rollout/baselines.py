"""Univariate short-term baselines for module eigengene trajectories.

Each baseline is a parameter-free, per-treatment/per-module rule. The days used
to *fit* the rule are configurable via ``fit_days`` (default ``1D/4D/8D``); the
remaining trajectory days are the prediction targets. With the default
``fit_days=(1, 4, 8)`` the rules predict the ``15D`` (validation) and ``29D``
(test) days. Passing ``fit_days=(1, 4, 8, 15)`` folds the 15D observations into
the fit so only ``29D`` is held out — every rule then reflects the data points
it actually consumed, e.g. ``"Early mean (1D4D8D15D)"``.

The leakage-safe :class:`~.scaling.FeatureScaler` is always fit on the first
three model days (1D/4D/8D); changing ``fit_days`` never changes standardization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import ModuleTrajectoryMatrix
from .evaluation import build_module_metric_tables, write_prediction_archive
from .reporting import (
    EXPECTED_REPLICATES_PER_TREATMENT_DAY,
    format_artifact_section,
    format_baseline_data_section,
    format_metric_section,
    format_parameter_section,
)
from .scaling import FeatureScaler


DAY_TO_INDEX: dict[int, int] = {1: 0, 4: 1, 8: 2, 15: 3, 29: 4}
#: All trajectory days in canonical order.
ALL_DAYS: tuple[int, ...] = (1, 4, 8, 15, 29)
#: Default fit days (1D/4D/8D); 15D validation and 29D test remain held out.
DEFAULT_FIT_DAYS: tuple[int, ...] = (1, 4, 8)
#: Human-readable day token, e.g. ``15 -> "15D"``.
DAY_TOKEN: dict[int, str] = {1: "1D", 4: "4D", 8: "8D", 15: "15D", 29: "29D"}

#: Short prose description of each prediction rule (keyed by rule slug).
BASELINE_RULE_DESCRIPTIONS: dict[str, str] = {
    "persistence": "prediction = module score at the latest fit day",
    "early_mean": "prediction = mean of module scores over all fit days",
    "linear_trend": "module score = intercept + slope * day (OLS over fit days)",
    "log_time_trend": "module score = intercept + slope * ln(day) (OLS over fit days)",
}


def _day_label(days: Sequence[int]) -> str:
    """Join day numbers into a compact token such as ``"1D4D8D15D"``.

    Args:
        days: Ordered day numbers drawn from :data:`ALL_DAYS`.

    Returns:
        Concatenated ``DAY``-suffixed tokens.
    """

    return "".join(DAY_TOKEN[day] for day in days)


def _baseline_names(fit_days: Sequence[int]) -> dict[str, str]:
    """Build display names for the four rules from their fit days.

    Args:
        fit_days: Days the rules consume (e.g. ``(1, 4, 8, 15)``).

    Returns:
        Rule slug -> human-readable label. ``persistence`` anchors on the latest
        fit day and reports only that day; the other three report every fit day.
    """

    fit_label = _day_label(fit_days)
    anchor_label = _day_label((max(fit_days),))
    return {
        "persistence": f"Persistence ({anchor_label})",
        "early_mean": f"Early mean ({fit_label})",
        "linear_trend": f"Linear trend ({fit_label})",
        "log_time_trend": f"Log-time trend ({fit_label})",
    }


def _split_by_day(predict_days: Sequence[int]) -> dict[int, str]:
    """Map each prediction day to its split role.

    Args:
        predict_days: Held-out days in canonical order (e.g. ``(15, 29)``).

    Returns:
        Day -> ``"validation"`` for every non-final prediction day and
        ``"test"`` for the final (latest) prediction day.
    """

    if not predict_days:
        raise ValueError("predict_days must contain at least one held-out day")
    latest = max(predict_days)
    return {day: ("test" if day == latest else "validation") for day in predict_days}


def persistence_predictions(
    values: np.ndarray,
    fit_days: Sequence[int],
    predict_days: Sequence[int],
    day_to_index: Mapping[int, int],
) -> dict[int, np.ndarray]:
    """Use each module's latest fit-day value for the held-out days.

    Args:
        values: Treatment-by-day-by-module trajectories.
        fit_days: Days the rule consumes; the latest is the persistence anchor.
        predict_days: Days to predict.
        day_to_index: Day-to-target-axis mapping.

    Returns:
        Predictions keyed by predicted day.
    """

    anchor = values[:, day_to_index[max(fit_days)]].copy()
    return {day: anchor.copy() for day in predict_days}


def early_mean_predictions(
    values: np.ndarray,
    fit_days: Sequence[int],
    predict_days: Sequence[int],
    day_to_index: Mapping[int, int],
) -> dict[int, np.ndarray]:
    """Use each module's mean over the fit days for the held-out days.

    Args:
        values: Treatment-by-day-by-module trajectories.
        fit_days: Days averaged into the early-mean constant.
        predict_days: Days to predict.
        day_to_index: Day-to-target-axis mapping.

    Returns:
        Predictions keyed by predicted day.
    """

    fit_indices = [day_to_index[day] for day in fit_days]
    mean = values[:, fit_indices].mean(axis=1)
    return {day: mean.copy() for day in predict_days}


def _trend_predictions(
    values: np.ndarray,
    fit_days: Sequence[int],
    predict_days: Sequence[int],
    day_to_index: Mapping[int, int],
    log_time: bool,
) -> dict[int, np.ndarray]:
    """Fit independent per-treatment/per-module OLS trends over the fit days.

    Args:
        values: Treatment-by-day-by-module trajectories.
        fit_days: Days regressed on (independent variable basis).
        predict_days: Days to evaluate the fitted trend at.
        day_to_index: Day-to-target-axis mapping.
        log_time: Whether to regress against natural-log day.

    Returns:
        Predictions keyed by predicted day.
    """

    x = np.log(np.asarray(fit_days, dtype=float)) if log_time else np.asarray(
        fit_days, dtype=float
    )
    design = np.column_stack([np.ones_like(x), x])
    fit_indices = [day_to_index[day] for day in fit_days]
    responses = values[:, fit_indices].transpose(1, 0, 2).reshape(len(fit_days), -1)
    coefficients, _, _, _ = np.linalg.lstsq(design, responses, rcond=None)
    predictions: dict[int, np.ndarray] = {}
    for day in predict_days:
        value = np.log(float(day)) if log_time else float(day)
        predictions[day] = (
            coefficients[0] + value * coefficients[1]
        ).reshape(values.shape[0], values.shape[2])
    return predictions


def linear_trend_predictions(
    values: np.ndarray,
    fit_days: Sequence[int],
    predict_days: Sequence[int],
    day_to_index: Mapping[int, int],
) -> dict[int, np.ndarray]:
    """Fit independent linear-time module trends over the fit days.

    Args:
        values: Treatment-by-day-by-module trajectories.
        fit_days: Days regressed on.
        predict_days: Days to predict.
        day_to_index: Day-to-target-axis mapping.

    Returns:
        Predictions keyed by predicted day.
    """

    return _trend_predictions(values, fit_days, predict_days, day_to_index, log_time=False)


def log_time_trend_predictions(
    values: np.ndarray,
    fit_days: Sequence[int],
    predict_days: Sequence[int],
    day_to_index: Mapping[int, int],
) -> dict[int, np.ndarray]:
    """Fit independent natural-log-time module trends over the fit days.

    Args:
        values: Treatment-by-day-by-module trajectories.
        fit_days: Days regressed on.
        predict_days: Days to predict.
        day_to_index: Day-to-target-axis mapping.

    Returns:
        Predictions keyed by predicted day.
    """

    return _trend_predictions(values, fit_days, predict_days, day_to_index, log_time=True)


def _resolve_fit_predict_days(
    fit_days: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate fit days and derive the held-out prediction days.

    Args:
        fit_days: Requested fit days (subset of :data:`ALL_DAYS`).

    Returns:
        ``(fit_days, predict_days)`` tuples in canonical order. ``predict_days``
        is every trajectory day not selected for fitting.
    """

    fit_set = set(fit_days)
    invalid = fit_set - set(ALL_DAYS)
    if invalid:
        raise ValueError(f"unsupported fit days {sorted(invalid)}; choose from {ALL_DAYS}")
    if not fit_set:
        raise ValueError("fit_days must contain at least one day")
    fit_ordered = tuple(day for day in ALL_DAYS if day in fit_set)
    predict_ordered = tuple(day for day in ALL_DAYS if day not in fit_set)
    if not predict_ordered:
        raise ValueError("fit_days covers every trajectory day; leave at least one held out")
    return fit_ordered, predict_ordered


def run_baselines(
    trajectories: ModuleTrajectoryMatrix,
    run_dir: Path,
    fit_days: Sequence[int] = DEFAULT_FIT_DAYS,
) -> pd.DataFrame:
    """Evaluate and write all four module baselines.

    Args:
        trajectories: Prepared module trajectories.
        run_dir: Empty output directory.
        fit_days: Days the baselines consume (default ``1D/4D/8D``). Every other
            trajectory day becomes a prediction target; the latest target is the
            test split and any earlier targets are validation.

    Returns:
        Global module metric table.
    """

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_ordered, predict_ordered = _resolve_fit_predict_days(fit_days)
    split_by_day = _split_by_day(predict_ordered)
    names = _baseline_names(fit_ordered)
    definitions = {names[rule]: BASELINE_RULE_DESCRIPTIONS[rule] for rule in names}

    module_scaler = FeatureScaler.fit(trajectories.values)
    targets_std = module_scaler.transform(trajectories.values)

    def _bind(function):
        return lambda values: function(
            values, fit_ordered, predict_ordered, DAY_TO_INDEX
        )

    functions = {
        names["persistence"]: _bind(persistence_predictions),
        names["early_mean"]: _bind(early_mean_predictions),
        names["linear_trend"]: _bind(linear_trend_predictions),
        names["log_time_trend"]: _bind(log_time_trend_predictions),
    }
    predictions_raw = {name: function(trajectories.values) for name, function in functions.items()}
    predictions_std = {
        name: {day: module_scaler.transform(values) for day, values in prediction.items()}
        for name, prediction in predictions_raw.items()
    }
    metrics, treatment_metrics, module_metrics = build_module_metric_tables(
        predictions_raw,
        predictions_std,
        trajectories.values,
        targets_std,
        trajectories.treatment_ids,
        trajectories.module_ids,
        DAY_TO_INDEX,
        split_by_day,
    )
    write_prediction_archive(
        run_dir / "predictions.npz",
        predictions_raw,
        predictions_std,
        trajectories.values,
        targets_std,
        trajectories.treatment_ids,
        trajectories.module_ids,
        DAY_TO_INDEX,
    )
    metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
    treatment_metrics.to_csv(run_dir / "treatment_metrics.tsv", sep="\t", index=False)
    module_metrics.to_csv(run_dir / "module_metrics.tsv", sep="\t", index=False)
    validation_days = [day for day, split in split_by_day.items() if split == "validation"]
    config: dict[str, object] = {
        "schema_version": 3,
        "pipeline": "module_dynamics/basic_rollout",
        "run_type": "baseline",
        "organ": trajectories.scope.organ,
        "scope_tag": trajectories.scope.scope_tag,
        "is_compound_scope": trajectories.scope.is_compound_scope,
        "reduction_config_id": trajectories.scope.config_id,
        "split_id": trajectories.scope.split_id,
        "administration_route": "Gavage",
        "models": list(functions),
        "model_count": len(functions),
        "model_definitions": definitions,
        "fit_days": [DAY_TOKEN[day] for day in fit_ordered],
        "prediction_days": [DAY_TOKEN[day] for day in predict_ordered],
        "input_days": [DAY_TOKEN[day] for day in fit_ordered],
        "validation_day": (
            DAY_TOKEN[validation_days[0]] if len(validation_days) == 1
            else [DAY_TOKEN[day] for day in validation_days]
        ) if validation_days else None,
        "test_day": DAY_TOKEN[predict_ordered[-1]],
        "scaler_fit_days": ["1D", "4D", "8D"],
        "split_axis": "timepoint",
        "same_treatments_across_splits": True,
        "prediction_granularity": "treatment_x_module",
        "each_treatment_module_is_fit_independently": True,
        "uses_other_modules_or_treatments": False,
        "uses_15d_to_predict_29d": 15 in fit_ordered,
        "n_treatments": len(trajectories.treatment_ids),
        "n_modules": len(trajectories.module_ids),
        "expected_replicates_per_treatment_day": EXPECTED_REPLICATES_PER_TREATMENT_DAY,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Module Dynamics Baseline Summary",
        "",
        *format_parameter_section(config),
        "",
        *format_baseline_data_section(trajectories, fit_ordered, predict_ordered),
        "",
        *format_artifact_section(),
        "",
        *format_metric_section("Module-level evaluation metrics:", metrics),
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "ALL_DAYS",
    "BASELINE_RULE_DESCRIPTIONS",
    "DAY_TOKEN",
    "DAY_TO_INDEX",
    "DEFAULT_FIT_DAYS",
    "early_mean_predictions",
    "linear_trend_predictions",
    "log_time_trend_predictions",
    "persistence_predictions",
    "run_baselines",
]
