"""Load five-day module eigengene trajectories from a reduction config dir."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ..common import (
    BASIC_MODEL_DAYS,
    EIGHT_TIMEPOINT_HOURS,
    EXPECTED_REPLICATES,
    ReductionScope,
    eight_timepoint_view,
    five_day_view,
    load_reduction_trajectories,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModuleTrajectoryMatrix:
    """Five-day module trajectories for basic rollout.

    Args:
        scope: Parsed reduction scope.
        treatment_ids: Ordered treatment identifiers.
        module_ids: Ordered non-grey module identifiers.
        values: Treatment-by-day-by-module replicate means.
        replicate_values: Treatment-by-day-by-replicate-by-module scores.
        treatment_metadata: One metadata row per treatment.
    """

    scope: ReductionScope
    treatment_ids: list[str]
    module_ids: list[str]
    values: np.ndarray
    replicate_values: np.ndarray
    treatment_metadata: pd.DataFrame


def _sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module_trajectory_matrix(
    reduction_dir: Path,
    eight_timepoint: bool = False,
) -> ModuleTrajectoryMatrix:
    """Load and validate the module trajectory view.

    Args:
        reduction_dir: Reduction config directory.
        eight_timepoint: When ``True``, keep the full eight-timepoint axis
            (3H/6H/9H/1D/4D/8D/15D/29D) for the ``lrd_3h`` dynamics mode;
            otherwise subsample to the five repeat-phase days (default).

    Returns:
        Dense module trajectory matrix (treatment × timepoint × module).
    """

    data = load_reduction_trajectories(reduction_dir)
    if eight_timepoint:
        means, replicates, _ = eight_timepoint_view(data)
        expected = len(EIGHT_TIMEPOINT_HOURS)
    else:
        means, replicates, _ = five_day_view(data)
        expected = len(BASIC_MODEL_DAYS)
    if means.shape[1] != expected:
        raise ValueError(
            f"trajectory view did not select {expected} timepoints "
            f"(got {means.shape[1]})"
        )
    if not np.isfinite(means).all() or not np.isfinite(replicates).all():
        raise ValueError("module trajectory contains non-finite values")
    return ModuleTrajectoryMatrix(
        scope=data.scope,
        treatment_ids=data.treatment_ids,
        module_ids=data.module_ids,
        values=means.astype(np.float64),
        replicate_values=replicates.astype(np.float64),
        treatment_metadata=data.treatment_metadata,
    )


def build_dataset_metadata(
    matrix: ModuleTrajectoryMatrix,
    reduction_dir: Path,
) -> dict[str, object]:
    """Build the serializable dataset contract for a prepared cache.

    Args:
        matrix: Loaded five-day module trajectory matrix.
        reduction_dir: Source reduction config directory.

    Returns:
        JSON-compatible dataset metadata.
    """

    trajectory_path = reduction_dir / "projection" / "treatment_module_trajectories.npz"
    try:
        display_reduction_dir = reduction_dir.resolve().relative_to(PROJECT_ROOT.resolve())
        display_trajectory_path = trajectory_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        display_reduction_dir = Path("<reduction-dir>")
        display_trajectory_path = Path("<reduction-dir>/projection/treatment_module_trajectories.npz")
    return {
        "schema_version": 1,
        "pipeline": "module_dynamics/basic_rollout",
        "organ": matrix.scope.organ,
        "scope_tag": matrix.scope.scope_tag,
        "is_compound_scope": matrix.scope.is_compound_scope,
        "reduction_run": matrix.scope.run_name,
        "split_id": matrix.scope.split_id,
        "reduction_config_id": matrix.scope.config_id,
        "model_days": list(BASIC_MODEL_DAYS),
        "expected_replicates": EXPECTED_REPLICATES,
        "n_treatments": len(matrix.treatment_ids),
        "n_modules": len(matrix.module_ids),
        "module_score_definition": "train-standardized fixed WGCNA module PC1 score",
        "inputs": {
            "reduction_dir": display_reduction_dir.as_posix(),
            "trajectory_npz": display_trajectory_path.as_posix(),
            "trajectory_npz_sha256": _sha256(trajectory_path) if trajectory_path.is_file() else "",
        },
    }


def write_module_cache(
    matrix: ModuleTrajectoryMatrix,
    reduction_dir: Path,
    output_dir: Path,
) -> None:
    """Write a prepared module cache without overwriting existing artifacts.

    Args:
        matrix: Loaded module trajectory matrix.
        reduction_dir: Source reduction config directory.
        output_dir: Empty output directory.

    Raises:
        FileExistsError: If any target artifact already exists.
    """

    targets = {
        "trajectory_npz": output_dir / "treatment_module_trajectories.npz",
        "treatment_meta": output_dir / "treatment_metadata.tsv",
        "metadata": output_dir / "dataset_metadata.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Prepared outputs already exist; remove them explicitly before rebuilding: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        targets["trajectory_npz"],
        values=matrix.values,
        replicate_values=matrix.replicate_values,
        treatment_ids=np.asarray(matrix.treatment_ids, dtype=str),
        model_days=np.asarray(BASIC_MODEL_DAYS, dtype=str),
        module_ids=np.asarray(matrix.module_ids, dtype=str),
    )
    matrix.treatment_metadata.to_csv(targets["treatment_meta"], sep="\t", index=False)
    metadata = build_dataset_metadata(matrix, reduction_dir)
    targets["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "PROJECT_ROOT",
    "ModuleTrajectoryMatrix",
    "build_dataset_metadata",
    "load_module_trajectory_matrix",
    "write_module_cache",
]
