"""Evaluation metrics and statistical comparators used in the manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


@dataclass(frozen=True)
class Metrics:
    """Mean squared error and Pearson correlation."""

    mse: float
    pearson_r: float
    n_values: int


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Metrics:
    """Calculate manuscript endpoint metrics after flattening all values.

    Args:
        observed: Observed values.
        predicted: Predictions with the same shape.

    Returns:
        MSE, Pearson correlation, and flattened value count.
    """

    obs = np.asarray(observed, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    if obs.shape != pred.shape or obs.size == 0:
        raise ValueError("observed and predicted must have the same non-empty shape")
    if not np.isfinite(obs).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require finite values")
    correlation = float(pearsonr(obs, pred).statistic)
    return Metrics(
        mse=float(np.mean((pred - obs) ** 2)),
        pearson_r=correlation,
        n_values=int(obs.size),
    )


def evaluate_splits(
    times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Evaluate the training, validation, and test time splits.

    Args:
        times: Observation times in days.
        observed: Dose by time by replicate by module observations, or condition means.
        predicted: Dose by time by module predictions, or an array matching ``observed``.

    Returns:
        One row per temporal split.
    """

    observed_values = np.asarray(observed, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if observed_values.ndim == 4 and predicted_values.ndim == 3:
        predicted_values = np.repeat(predicted_values[:, :, None, :], observed_values.shape[2], axis=2)
    if observed_values.shape != predicted_values.shape:
        raise ValueError("observed and predicted axes are incompatible")
    split_masks = {
        "training": np.asarray(times) <= 8.0,
        "validation": np.isclose(times, 15.0),
        "test": np.isclose(times, 29.0),
    }
    rows: list[dict[str, float | int | str]] = []
    for split_name, mask in split_masks.items():
        metric = calculate_metrics(observed_values[:, mask, ...], predicted_values[:, mask, ...])
        rows.append(
            {
                "split": split_name,
                "mse": metric.mse,
                "pearson_r": metric.pearson_r,
                "n_values": metric.n_values,
            }
        )
    return pd.DataFrame(rows)


def _trend_prediction(
    fit_times: np.ndarray,
    fit_values: np.ndarray,
    predict_times: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Fit independent OLS trends for every dose-module trajectory."""

    x_fit = transform(fit_times)
    design = np.column_stack([np.ones_like(x_fit), x_fit])
    responses = fit_values.transpose(1, 0, 2).reshape(len(fit_times), -1)
    coefficients, _, _, _ = np.linalg.lstsq(design, responses, rcond=None)
    outputs: list[np.ndarray] = []
    for time in transform(predict_times):
        values = coefficients[0] + time * coefficients[1]
        outputs.append(values.reshape(fit_values.shape[0], fit_values.shape[2]))
    return np.stack(outputs, axis=1)


def statistical_baselines(
    times: np.ndarray,
    mean_values: np.ndarray,
    predict_days: Sequence[float] = (15.0, 29.0),
) -> dict[str, np.ndarray]:
    """Predict held-out endpoints with the four manuscript baselines.

    Args:
        times: Complete observation times in days.
        mean_values: Dose by time by module condition means.
        predict_days: Held-out days to predict.

    Returns:
        Baseline name to dose by predicted-time by module array.
    """

    fit_days = np.asarray([1.0, 4.0, 8.0])
    fit_indices = [int(np.flatnonzero(np.isclose(times, day))[0]) for day in fit_days]
    fit_values = mean_values[:, fit_indices, :]
    predictions = np.asarray(predict_days, dtype=float)
    constant_shape = (mean_values.shape[0], len(predictions), mean_values.shape[2])
    persistence = np.broadcast_to(fit_values[:, -1:, :], constant_shape).copy()
    early_mean = np.broadcast_to(fit_values.mean(axis=1, keepdims=True), constant_shape).copy()
    return {
        "early_mean": early_mean,
        "linear_trend": _trend_prediction(
            fit_days, fit_values, predictions, lambda values: values
        ),
        "log_time_trend": _trend_prediction(
            fit_days, fit_values, predictions, np.log
        ),
        "persistence": persistence,
    }


def evaluate_statistical_baselines(
    times: np.ndarray,
    mean_values: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all statistical comparators at day 15 and day 29.

    Args:
        times: Complete observation times in days.
        mean_values: Dose by time by module condition means.

    Returns:
        Long-form metric table.
    """

    predict_days = (15.0, 29.0)
    observed_indices = [int(np.flatnonzero(np.isclose(times, day))[0]) for day in predict_days]
    observed = mean_values[:, observed_indices, :]
    rows: list[dict[str, float | int | str]] = []
    for model_name, predictions in statistical_baselines(times, mean_values).items():
        for time_index, day in enumerate(predict_days):
            metric = calculate_metrics(
                observed[:, time_index, :], predictions[:, time_index, :]
            )
            rows.append(
                {
                    "model": model_name,
                    "day": int(day),
                    "mse": metric.mse,
                    "pearson_r": metric.pearson_r,
                    "n_values": metric.n_values,
                }
            )
    return pd.DataFrame(rows)
