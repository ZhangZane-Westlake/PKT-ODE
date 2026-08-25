"""Metrics and compact prediction exports for module rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Pearson correlation with constant-vector handling.

    Args:
        first: First one-dimensional vector.
        second: Second one-dimensional vector.

    Returns:
        Pearson correlation or NaN.
    """

    if first.size < 2 or second.size < 2:
        return float("nan")
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def summarize_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate flat error and correlation metrics.

    Args:
        target: Target array.
        prediction: Equal-shaped prediction array.

    Returns:
        MSE, RMSE, MAE, and flattened Pearson.
    """

    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have equal shapes")
    error = prediction - target
    mse = float(np.mean(np.square(error)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "pearson": safe_pearson(target.ravel(), prediction.ravel()),
    }


def build_module_metric_tables(
    model_predictions_raw: Mapping[str, Mapping[int, np.ndarray]],
    model_predictions_std: Mapping[str, Mapping[int, np.ndarray]],
    targets_raw: np.ndarray,
    targets_std: np.ndarray,
    treatment_ids: Sequence[str],
    module_ids: Sequence[str],
    day_to_index: Mapping[int, int],
    split_by_day: Mapping[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build global, treatment, and module metric tables.

    Args:
        model_predictions_raw: Raw predictions by model and day.
        model_predictions_std: Standardized predictions by model and day.
        targets_raw: Full raw treatment trajectories.
        targets_std: Full standardized treatment trajectories.
        treatment_ids: Ordered treatments.
        module_ids: Ordered modules.
        day_to_index: Day-to-target-axis mapping.
        split_by_day: Day-to-split mapping.

    Returns:
        Summary, treatment, and module metric tables.
    """

    summary_rows: list[dict[str, object]] = []
    treatment_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    for model, predictions_raw in model_predictions_raw.items():
        predictions_std = model_predictions_std[model]
        for day, prediction_raw in predictions_raw.items():
            target_raw = targets_raw[:, day_to_index[day]]
            target_std = targets_std[:, day_to_index[day]]
            prediction_std = predictions_std[day]
            for scale, target, prediction in (
                ("raw", target_raw, prediction_raw),
                ("standardized", target_std, prediction_std),
            ):
                row: dict[str, object] = {
                    "model": model,
                    "split": split_by_day[day],
                    "day": day,
                    "scale": scale,
                    "n_treatments": len(treatment_ids),
                    "n_modules": len(module_ids),
                }
                row.update(summarize_metrics(target, prediction))
                summary_rows.append(row)
            for index, treatment_id in enumerate(treatment_ids):
                metrics = summarize_metrics(target_raw[index], prediction_raw[index])
                treatment_rows.append(
                    {
                        "model": model,
                        "split": split_by_day[day],
                        "day": day,
                        "treatment_id": treatment_id,
                        **metrics,
                    }
                )
            for index, module_id in enumerate(module_ids):
                metrics = summarize_metrics(target_raw[:, index], prediction_raw[:, index])
                module_rows.append(
                    {
                        "model": model,
                        "split": split_by_day[day],
                        "day": day,
                        "module_id": module_id,
                        **metrics,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(treatment_rows), pd.DataFrame(module_rows)


def write_prediction_archive(
    output_path: Path,
    model_predictions_raw: Mapping[str, Mapping[int, np.ndarray]],
    model_predictions_std: Mapping[str, Mapping[int, np.ndarray]],
    targets_raw: np.ndarray,
    targets_std: np.ndarray,
    treatment_ids: Sequence[str],
    module_ids: Sequence[str],
    day_to_index: Mapping[int, int],
) -> None:
    """Write a no-pickle compressed prediction archive.

    Args:
        output_path: NPZ destination.
        model_predictions_raw: Raw predictions by model/day.
        model_predictions_std: Standardized predictions by model/day.
        targets_raw: Full raw trajectories.
        targets_std: Full standardized trajectories.
        treatment_ids: Ordered treatments.
        module_ids: Ordered modules.
        day_to_index: Day-to-target-axis mapping.
    """

    model_labels = list(model_predictions_raw)
    days = sorted(next(iter(model_predictions_raw.values())))
    for model in model_labels:
        if sorted(model_predictions_raw[model]) != days:
            raise ValueError("All models must provide the same prediction days")
    target_raw = np.stack([targets_raw[:, day_to_index[day]] for day in days], axis=0)
    target_std = np.stack([targets_std[:, day_to_index[day]] for day in days], axis=0)
    prediction_raw = np.stack(
        [np.stack([model_predictions_raw[model][day] for day in days]) for model in model_labels]
    )
    prediction_std = np.stack(
        [np.stack([model_predictions_std[model][day] for day in days]) for model in model_labels]
    )
    np.savez_compressed(
        output_path,
        model_labels=np.asarray(model_labels, dtype=str),
        days=np.asarray(days, dtype=np.int16),
        treatment_ids=np.asarray(treatment_ids, dtype=str),
        module_ids=np.asarray(module_ids, dtype=str),
        target_raw=target_raw,
        prediction_raw=prediction_raw,
        target_z=target_std,
        prediction_z=prediction_std,
    )


__all__ = [
    "build_module_metric_tables",
    "safe_pearson",
    "summarize_metrics",
    "write_prediction_archive",
]
