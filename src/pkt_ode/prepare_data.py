"""Build the compact, MAD-normalized input used by PKT-ODE."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

import numpy as np
import pandas as pd


MAD_TO_STD: Final[float] = 1.4826


def _dose_from_treatment_id(treatment_id: str) -> float:
    """Extract the numeric dose from one portable treatment identifier."""

    match = re.search(r"DOSE-([0-9]+(?:\.[0-9]+)?)-MG-KG", treatment_id)
    if match is None:
        raise ValueError(f"Cannot parse dose from treatment identifier: {treatment_id}")
    return float(match.group(1))


def build_pkt_ode_input(
    trajectory_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> None:
    """Create the aligned PKT-ODE archive from frozen projection artifacts.

    Args:
        trajectory_path: Standardized treatment trajectory archive.
        bundle_path: Frozen fixed-PC1 projection bundle.
        metadata_path: Sample metadata table aligned to the projection.
        output_path: Destination for the compressed PKT-ODE archive.

    Raises:
        ValueError: If axes, metadata, or normalization scales are inconsistent.
    """

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        standardized = np.asarray(trajectory["replicate_values"], dtype=float)
        treatment_ids = tuple(str(value) for value in trajectory["treatment_ids"])
        model_times = tuple(str(value) for value in trajectory["model_times"])
        module_ids = np.asarray(trajectory["module_ids"], dtype=str)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        bundle_module_ids = np.asarray(bundle["module_ids"], dtype=str)
        module_center = np.asarray(bundle["module_center"], dtype=float)
        module_scale = np.asarray(bundle["module_scale"], dtype=float)
    if not np.array_equal(module_ids, bundle_module_ids):
        raise ValueError("Trajectory and projection-bundle module axes differ")
    if standardized.shape != (len(treatment_ids), len(model_times), 3, len(module_ids)):
        raise ValueError("Trajectory archive does not have the expected four axes")

    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    required_columns = {
        "sample_id",
        "treatment_id",
        "model_time",
        "time_hours",
        "dose_level",
        "regimen",
        "used_for_reduction_fit",
    }
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    row_blocks: list[pd.DataFrame] = []
    standardized_blocks: list[np.ndarray] = []
    for treatment_index, treatment_id in enumerate(treatment_ids):
        for time_index, model_time in enumerate(model_times):
            rows = metadata[
                (metadata["treatment_id"].astype(str) == treatment_id)
                & (metadata["model_time"].astype(str) == model_time)
            ]
            if len(rows) != 3:
                raise ValueError(
                    f"Expected three samples for {treatment_id} at {model_time}, got {len(rows)}"
                )
            row_blocks.append(rows)
            standardized_blocks.append(standardized[treatment_index, time_index])

    aligned_metadata = pd.concat(row_blocks, ignore_index=True)
    standardized_rows = np.concatenate(standardized_blocks, axis=0)
    raw_rows = standardized_rows * module_scale + module_center
    fit_mask = aligned_metadata["used_for_reduction_fit"].astype(bool).to_numpy()
    fit_raw = raw_rows[fit_mask]
    if fit_raw.shape[0] != 54:
        raise ValueError(f"Expected 54 fit-window profiles, got {fit_raw.shape[0]}")
    medians = np.median(fit_raw, axis=0)
    mad_fit = np.median(np.abs(fit_raw - medians), axis=0) * MAD_TO_STD
    std_fit = np.std(fit_raw, axis=0)
    if np.any(mad_fit <= 0.0) or not np.isfinite(mad_fit).all():
        raise ValueError("Fit-window MAD scales must be positive and finite")
    normalized_rows = raw_rows / mad_fit

    doses = np.asarray(
        [_dose_from_treatment_id(str(value)) for value in aligned_metadata["treatment_id"]],
        dtype=float,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([1], dtype=np.int64),
        raw_values=raw_rows.astype(np.float32),
        normalized_values=normalized_rows.astype(np.float32),
        mad_fit=mad_fit.astype(np.float32),
        std_fit=std_fit.astype(np.float32),
        sample_ids=aligned_metadata["sample_id"].astype(str).to_numpy(dtype=str),
        time_hours=aligned_metadata["time_hours"].to_numpy(dtype=float),
        dose=doses,
        dose_level=aligned_metadata["dose_level"].astype(str).to_numpy(dtype=str),
        regimen=aligned_metadata["regimen"].astype(str).to_numpy(dtype=str),
        used_for_fit=fit_mask,
        module_ids=module_ids,
        normalization=np.asarray(["fit_window_mad_x_1.4826"], dtype=str),
    )
