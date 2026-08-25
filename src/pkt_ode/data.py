"""Typed readers for the compact public PKT-ODE data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .model import ModuleParameters


TIME_IN_DAYS: Mapping[str, float] = {
    "3H": 0.125,
    "6H": 0.25,
    "9H": 0.375,
    "1D": 1.0,
    "4D": 4.0,
    "8D": 8.0,
    "15D": 15.0,
    "29D": 29.0,
}


@dataclass(frozen=True)
class TrajectoryData:
    """Observed MAD-normalized directed-PC1 module trajectories.

    Attributes:
        replicate_values: Dose by time by replicate by module MAD-normalized values.
        raw_replicate_values: Corresponding unnormalized directed-PC1 values.
        mean_values: Dose by time by module condition means.
        sd_values: Dose by time by module sample standard deviations.
        doses: Daily administered doses in mg/kg.
        times: Observation times in days.
        time_labels: Display labels corresponding to ``times``.
        module_ids: Ordered module identifiers.
        regimens: Single- or repeat-dose label for each observation time.
        sample_ids: Dose by time by replicate public sample identifiers.
        used_for_fit: Dose by time by replicate fit-window indicators.
        mad_fit: Robust fit-window normalization scale for each module.
        std_fit: Conventional fit-window standard deviation for each module.
    """

    replicate_values: np.ndarray
    raw_replicate_values: np.ndarray
    mean_values: np.ndarray
    sd_values: np.ndarray
    doses: np.ndarray
    times: np.ndarray
    time_labels: tuple[str, ...]
    module_ids: tuple[str, ...]
    regimens: tuple[str, ...]
    sample_ids: np.ndarray
    used_for_fit: np.ndarray
    mad_fit: np.ndarray
    std_fit: np.ndarray


def load_trajectories(path: Path) -> TrajectoryData:
    """Load and validate the compact trajectory archive.

    Args:
        path: Path to ``pkt_ode_input.npz``.

    Returns:
        Validated trajectory data in ascending dose order.

    Raises:
        ValueError: If required arrays, axes, or finite values are invalid.
    """

    required = {
        "raw_values",
        "normalized_values",
        "mad_fit",
        "std_fit",
        "sample_ids",
        "time_hours",
        "dose",
        "regimen",
        "used_for_fit",
        "module_ids",
        "normalization",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Trajectory archive is missing arrays: {sorted(missing)}")
        raw_rows = np.asarray(archive["raw_values"], dtype=float)
        normalized_rows = np.asarray(archive["normalized_values"], dtype=float)
        sample_ids_rows = np.asarray(archive["sample_ids"], dtype=str)
        time_hours_rows = np.asarray(archive["time_hours"], dtype=float)
        dose_rows = np.asarray(archive["dose"], dtype=float)
        regimen_rows = np.asarray(archive["regimen"], dtype=str)
        fit_rows = np.asarray(archive["used_for_fit"], dtype=bool)
        module_ids = tuple(str(value) for value in archive["module_ids"])
        mad_fit = np.asarray(archive["mad_fit"], dtype=float)
        std_fit = np.asarray(archive["std_fit"], dtype=float)
        normalization = tuple(str(value) for value in archive["normalization"])

    expected_rows = 3 * len(TIME_IN_DAYS) * 3
    expected_value_shape = (expected_rows, len(module_ids))
    if raw_rows.shape != expected_value_shape or normalized_rows.shape != expected_value_shape:
        raise ValueError(f"PKT-ODE value matrices must have shape {expected_value_shape}")
    if any(values.shape != (expected_rows,) for values in (
        sample_ids_rows, time_hours_rows, dose_rows, regimen_rows, fit_rows
    )):
        raise ValueError("PKT-ODE sample metadata axes are inconsistent")
    if mad_fit.shape != (len(module_ids),) or std_fit.shape != (len(module_ids),):
        raise ValueError("PKT-ODE normalization scales do not match the module axis")
    if not all(np.isfinite(values).all() for values in (
        raw_rows, normalized_rows, time_hours_rows, dose_rows, mad_fit, std_fit
    )):
        raise ValueError("PKT-ODE archive contains non-finite values")
    if np.any(mad_fit <= 0.0) or np.any(std_fit <= 0.0):
        raise ValueError("PKT-ODE normalization scales must be positive")
    if normalization != ("fit_window_mad_x_1.4826",):
        raise ValueError(f"Unsupported PKT-ODE normalization: {normalization}")

    doses = np.unique(dose_rows)
    times = np.unique(time_hours_rows) / 24.0
    if len(doses) != 3 or len(times) != len(TIME_IN_DAYS):
        raise ValueError("PKT-ODE archive must contain three doses and eight times")
    shape = (len(doses), len(times), 3, len(module_ids))
    replicate_values = np.empty(shape, dtype=float)
    raw_replicate_values = np.empty(shape, dtype=float)
    sample_ids = np.empty(shape[:-1], dtype=sample_ids_rows.dtype)
    used_for_fit = np.empty(shape[:-1], dtype=bool)
    regimens: list[str] = []
    for dose_index, dose in enumerate(doses):
        for time_index, time in enumerate(times):
            indices = np.flatnonzero(
                np.isclose(dose_rows, dose) & np.isclose(time_hours_rows / 24.0, time)
            )
            if len(indices) != 3:
                raise ValueError(f"Expected three profiles for dose={dose:g}, time={time:g}")
            unique_regimens = np.unique(regimen_rows[indices])
            if len(unique_regimens) != 1:
                raise ValueError("Each dose-time condition must have one regimen label")
            replicate_values[dose_index, time_index] = normalized_rows[indices]
            raw_replicate_values[dose_index, time_index] = raw_rows[indices]
            sample_ids[dose_index, time_index] = sample_ids_rows[indices]
            used_for_fit[dose_index, time_index] = fit_rows[indices]
            if dose_index == 0:
                regimens.append(str(unique_regimens[0]))
            elif regimens[time_index] != str(unique_regimens[0]):
                raise ValueError("Regimen labels must agree across doses")

    time_labels_by_day = {value: key for key, value in TIME_IN_DAYS.items()}
    unknown_times = [time for time in times if float(time) not in time_labels_by_day]
    if unknown_times:
        raise ValueError(f"Unknown observation times: {unknown_times}")
    expected_fit = np.broadcast_to(times[None, :, None] <= 8.0, used_for_fit.shape)
    if not np.array_equal(used_for_fit, expected_fit):
        raise ValueError("Fit-window flags must include only the 54 profiles through day 8")
    mean_values = replicate_values.mean(axis=2)
    sd_values = replicate_values.std(axis=2, ddof=1)
    return TrajectoryData(
        replicate_values=replicate_values,
        raw_replicate_values=raw_replicate_values,
        mean_values=mean_values,
        sd_values=sd_values,
        doses=doses,
        times=times,
        time_labels=tuple(time_labels_by_day[float(time)] for time in times),
        module_ids=module_ids,
        regimens=tuple(regimens),
        sample_ids=sample_ids,
        used_for_fit=used_for_fit,
        mad_fit=mad_fit,
        std_fit=std_fit,
    )


def load_module_parameters(path: Path) -> dict[str, ModuleParameters]:
    """Load the published per-module PKT-ODE parameters.

    Args:
        path: Tab-separated parameter table.

    Returns:
        Module identifier to parameter mapping.

    Raises:
        ValueError: If required columns are absent or values are duplicated.
    """

    table = pd.read_csv(path, sep="\t")
    required = {"module_id", "k_per_day", "beta_0", "beta_1"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Parameter table is missing columns: {sorted(missing)}")
    if table["module_id"].duplicated().any():
        raise ValueError("Parameter table contains duplicate module identifiers")
    return {
        str(row.module_id): ModuleParameters(
            k_per_day=float(row.k_per_day),
            beta_0=float(row.beta_0),
            beta_1=float(row.beta_1),
        )
        for row in table.itertuples(index=False)
    }
