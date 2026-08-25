"""Shared reduction-directory loading and scope parsing for comparators."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd

from ..gene_module_reduction.eligibility import (
    ELIGIBILITY_MODEL_TIMES,
    MODEL_TIME_HOURS,
)
from ..gene_module_reduction.projection import (
    BUNDLE_SCHEMA_VERSION,
    ModuleBundle,
    load_module_bundle,
)


BASIC_MODEL_DAYS: Final[tuple[str, ...]] = ("1D", "4D", "8D", "15D", "29D")
DAY_TO_INDEX: Final[dict[int, int]] = {1: 0, 4: 1, 8: 2, 15: 3, 29: 4}
SPLIT_BY_DAY: Final[dict[int, str]] = {
    4: "train",
    8: "train",
    15: "validation",
    29: "test",
}
#: Indices into the eight-timepoint axis that select the five repeat-phase days.
REPEAT_PHASE_INDICES: Final[tuple[int, ...]] = (3, 4, 5, 6, 7)
#: Single-dose animal phase times (3H/6H/9H/1D).
SINGLE_PHASE_TIMES: Final[tuple[str, ...]] = ("3H", "6H", "9H", "1D")
#: Repeat-dose animal phase times (4D/8D/15D/29D).
REPEAT_PHASE_TIMES: Final[tuple[str, ...]] = ("4D", "8D", "15D", "29D")

#: Full eight-timepoint axis in elapsed hours (3H/6H/9H/1D/4D/8D/15D/29D), used
#: by the basic_rollout ``lrd_3h`` dynamics mode (3-hour LRD granularity).
EIGHT_TIMEPOINT_HOURS: Final[tuple[int, ...]] = (3, 6, 9, 24, 96, 192, 360, 696)
#: Positional index of each elapsed hour within the eight-timepoint axis.
HOUR_TO_INDEX: Final[dict[int, int]] = {
    hour: idx for idx, hour in enumerate(EIGHT_TIMEPOINT_HOURS)
}
#: ``lrd_3h`` rollout contract: init at 3H, step 3h, train on the hours below.
LRD3H_INIT_HOUR: Final[int] = 3
LRD3H_STEP_HOURS: Final[int] = 3
LRD3H_TRAIN_HOURS: Final[tuple[int, ...]] = (6, 9, 24, 96, 192)  # 6H/9H/1D/4D/8D
LRD3H_VAL_HOUR: Final[int] = 360  # 15D
LRD3H_TEST_HOUR: Final[int] = 696  # 29D
#: Split role for each predicted hour on the ``lrd_3h`` hour axis.
SPLIT_BY_HOUR: Final[dict[int, str]] = {
    6: "train",
    9: "train",
    24: "train",
    96: "train",
    192: "train",
    360: "validation",
    696: "test",
}

REQUIRED_PROJECTION_FILES: Final[tuple[str, ...]] = (
    "treatment_module_trajectories.npz",
    "module_bundle.npz",
    "sample_metadata.tsv.gz",
)
TREATMENT_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "treatment_id",
    "compound_no",
    "compound_name",
    "dose",
    "dose_unit",
    "organ",
)
EXPECTED_REPLICATES: Final[int] = 3


@dataclass(frozen=True)
class ReductionScope:
    """Resolved scope parsed from a reduction config directory path.

    Args:
        organ: ``kidney`` or ``liver``.
        compound_slug: Compound directory slug for single-compound scopes.
        split_id: ``loose_stage`` or ``tight_stage``.
        config_id: WGCNA identity directory name.
        is_compound_scope: Whether the path is single-compound scoped.
        run_name: Reduction run name (e.g. ``gene_selection_20260731``).
    """

    organ: str
    compound_slug: Optional[str]
    split_id: str
    config_id: str
    is_compound_scope: bool
    run_name: str

    @property
    def scope_tag(self) -> str:
        """Return a filesystem-safe scope label for output routing.

        Returns:
            ``compounds/<slug>`` for compound scope, else the organ name.
        """

        if self.is_compound_scope:
            assert self.compound_slug is not None
            return f"compounds/{self.compound_slug}"
        return self.organ


def parse_reduction_scope(reduction_dir: Path) -> ReductionScope:
    """Parse a reduction config directory path into a scope.

    Supports both layouts:

    * compound scope:
      ``.../{run}/{organ}/compounds/{slug}/{split}/reductions/{config_id}``
    * organ scope:
      ``.../{run}/{organ}/{split}/reductions/{config_id}``

    Args:
        reduction_dir: Reduction config directory.

    Returns:
        Parsed scope.

    Raises:
        ValueError: If the path does not match either layout.
    """

    reduction_path = Path(reduction_dir)
    public_scope_path = reduction_path / "reduction_scope.json"
    if public_scope_path.is_file():
        scope = json.loads(public_scope_path.read_text(encoding="utf-8"))
        required = {
            "organ",
            "compound_slug",
            "split_id",
            "config_id",
            "is_compound_scope",
            "run_name",
        }
        missing = required.difference(scope)
        if missing:
            raise ValueError(f"reduction_scope.json is missing fields: {sorted(missing)}")
        return ReductionScope(
            organ=str(scope["organ"]),
            compound_slug=(
                str(scope["compound_slug"])
                if scope["compound_slug"] is not None
                else None
            ),
            split_id=str(scope["split_id"]),
            config_id=str(scope["config_id"]),
            is_compound_scope=bool(scope["is_compound_scope"]),
            run_name=str(scope["run_name"]),
        )

    parts = reduction_path.resolve().parts
    if len(parts) < 5 or parts[-2] != "reductions":
        raise ValueError(
            f"reduction-dir must end in reductions/<config_id>: {reduction_dir}"
        )
    config_id = parts[-1]
    split_id = parts[-3]
    if split_id not in {"loose_stage", "tight_stage"}:
        raise ValueError(f"unexpected split directory '{split_id}' in {reduction_dir}")
    if len(parts) >= 6 and parts[-5] == "compounds":
        organ = parts[-6]
        compound_slug = parts[-4]
        is_compound_scope = True
        run_name = parts[-7] if len(parts) >= 7 else ""
    else:
        organ = parts[-4]
        compound_slug = None
        is_compound_scope = False
        run_name = parts[-5] if len(parts) >= 5 else ""
    if organ not in {"kidney", "liver"}:
        raise ValueError(f"unexpected organ '{organ}' in {reduction_dir}")
    return ReductionScope(
        organ=organ,
        compound_slug=compound_slug,
        split_id=split_id,
        config_id=config_id,
        is_compound_scope=is_compound_scope,
        run_name=run_name,
    )


@dataclass(frozen=True)
class ModuleTrajectoryData:
    """Module eigengene trajectories loaded from one reduction config.

    Values are the train-standardized fixed PC1 scores written by
    ``gene_module_reduction``. The eight-timepoint axis preserves the original
    order ``3H, 6H, 9H, 1D, 4D, 8D, 15D, 29D``.

    Args:
        scope: Parsed reduction scope.
        treatment_ids: Ordered treatment identifiers.
        module_ids: Ordered non-grey module identifiers.
        model_times: Eight ordered model-time labels.
        mean_values: Treatment-by-time-by-module replicate means.
        replicate_values: Treatment-by-time-by-replicate-by-module scores.
        sd_values: Treatment-by-time-by-module sample standard deviations.
        treatment_metadata: One row per treatment with compound/dose/organ.
        sample_metadata: Full eligible-sample metadata (regimen, dose, hours).
        bundle: Fixed module projection bundle.
    """

    scope: ReductionScope
    treatment_ids: list[str]
    module_ids: list[str]
    model_times: list[str]
    mean_values: np.ndarray
    replicate_values: np.ndarray
    sd_values: np.ndarray
    treatment_metadata: pd.DataFrame
    sample_metadata: pd.DataFrame
    bundle: ModuleBundle


def _build_treatment_metadata(
    sample_metadata: pd.DataFrame,
    treatment_ids: list[str],
) -> pd.DataFrame:
    """Collapse per-sample metadata to one row per treatment.

    Args:
        sample_metadata: Full eligible-sample metadata.
        treatment_ids: Ordered treatment identifiers.

    Returns:
        Treatment-level metadata frame.
    """

    rows: list[dict[str, object]] = []
    for treatment_id in treatment_ids:
        subset = sample_metadata.loc[sample_metadata["treatment_id"].astype(str).eq(treatment_id)]
        if subset.empty:
            raise ValueError(f"treatment {treatment_id} has no sample metadata rows")
        first = subset.iloc[0]
        rows.append(
            {
                "treatment_id": treatment_id,
                "compound_no": str(first["compound_no"]).zfill(5),
                "compound_name": str(first["compound_name"]),
                "dose": str(first["dose"]),
                "dose_unit": str(first["dose_unit"]),
                "organ": str(first["organ"]).lower(),
            }
        )
    return pd.DataFrame(rows)


def load_reduction_trajectories(reduction_dir: Path) -> ModuleTrajectoryData:
    """Load and validate module trajectories from a reduction config dir.

    Args:
        reduction_dir: Reduction config directory containing ``projection/``.

    Returns:
        Loaded module trajectory data.

    Raises:
        FileNotFoundError: If a required projection artifact is missing.
        ValueError: If axes, shapes, or values are inconsistent.
    """

    reduction_dir = Path(reduction_dir)
    projection_dir = reduction_dir / "projection"
    for name in REQUIRED_PROJECTION_FILES:
        if not (projection_dir / name).is_file():
            raise FileNotFoundError(
                f"Missing projection artifact {name} under {projection_dir}"
            )
    scope = parse_reduction_scope(reduction_dir)
    bundle = load_module_bundle(projection_dir / "module_bundle.npz")
    trajectory_path = projection_dir / "treatment_module_trajectories.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        replicate_values = archive["replicate_values"].astype(np.float64, copy=False)
        mean_values = archive["mean_values"].astype(np.float64, copy=False)
        sd_values = archive["sd_values"].astype(np.float64, copy=False)
        treatment_ids = archive["treatment_ids"].astype(str).tolist()
        model_times = archive["model_times"].astype(str).tolist()
        module_ids = archive["module_ids"].astype(str).tolist()
    if tuple(model_times) != tuple(ELIGIBILITY_MODEL_TIMES):
        raise ValueError(f"unexpected model times: {model_times}")
    if module_ids != bundle.module_ids:
        raise ValueError("trajectory module axis does not match module bundle")
    n_treatments = len(treatment_ids)
    n_modules = len(module_ids)
    expected_shape = (
        n_treatments,
        len(ELIGIBILITY_MODEL_TIMES),
        EXPECTED_REPLICATES,
        n_modules,
    )
    if replicate_values.shape != expected_shape:
        raise ValueError(f"replicate_values shape {replicate_values.shape} != {expected_shape}")
    if mean_values.shape != (n_treatments, len(ELIGIBILITY_MODEL_TIMES), n_modules):
        raise ValueError(f"mean_values shape mismatch: {mean_values.shape}")
    if sd_values.shape != (n_treatments, len(ELIGIBILITY_MODEL_TIMES), n_modules):
        raise ValueError(f"sd_values shape mismatch: {sd_values.shape}")
    for name, values in (
        ("replicate_values", replicate_values),
        ("mean_values", mean_values),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    sample_metadata = pd.read_csv(
        projection_dir / "sample_metadata.tsv.gz", sep="\t", dtype=str
    )
    sample_metadata["time_hours"] = (
        sample_metadata["model_time"].map(MODEL_TIME_HOURS).astype(int)
    )
    treatment_metadata = _build_treatment_metadata(sample_metadata, treatment_ids)
    return ModuleTrajectoryData(
        scope=scope,
        treatment_ids=treatment_ids,
        module_ids=module_ids,
        model_times=list(model_times),
        mean_values=mean_values,
        replicate_values=replicate_values,
        sd_values=sd_values,
        treatment_metadata=treatment_metadata,
        sample_metadata=sample_metadata,
        bundle=bundle,
    )


def five_day_view(data: ModuleTrajectoryData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample the eight-timepoint axis to the five repeat-phase days.

    Args:
        data: Full eight-timepoint module trajectory data.

    Returns:
        Mean, replicate, and standard-deviation arrays on the five-day axis
        ``(1D, 4D, 8D, 15D, 29D)``.
    """

    indices = REPEAT_PHASE_INDICES
    means = data.mean_values[:, indices, :]
    replicates = data.replicate_values[:, indices, :, :]
    sds = data.sd_values[:, indices, :]
    return means, replicates, sds


def eight_timepoint_view(data: ModuleTrajectoryData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the full eight-timepoint axis (3H/6H/9H/1D/4D/8D/15D/29D).

    Unlike :func:`five_day_view`, this keeps the 3H/6H/9H short-term columns so
    the basic_rollout ``lrd_3h`` dynamics mode can learn from and predict the
    early responses. The input is unchanged; this is a passthrough kept for
    symmetry with the five-day view.

    Args:
        data: Full eight-timepoint module trajectory data.

    Returns:
        Mean, replicate, and standard-deviation arrays on the eight-hour axis.
    """

    return data.mean_values, data.replicate_values, data.sd_values


def module_axis_sha256(data: ModuleTrajectoryData) -> str:
    """Return the bundle module-axis digest for cross-reduction matching.

    Args:
        data: Loaded module trajectory data.

    Returns:
        Hexadecimal SHA-256 digest of the module axis.
    """

    import hashlib

    return hashlib.sha256(
        "\n".join(data.module_ids).encode("utf-8")
    ).hexdigest()


__all__ = [
    "BASIC_MODEL_DAYS",
    "BUNDLE_SCHEMA_VERSION",
    "DAY_TO_INDEX",
    "EIGHT_TIMEPOINT_HOURS",
    "EXPECTED_REPLICATES",
    "HOUR_TO_INDEX",
    "LRD3H_INIT_HOUR",
    "LRD3H_STEP_HOURS",
    "LRD3H_TEST_HOUR",
    "LRD3H_TRAIN_HOURS",
    "LRD3H_VAL_HOUR",
    "ModuleTrajectoryData",
    "ReductionScope",
    "SPLIT_BY_DAY",
    "SPLIT_BY_HOUR",
    "eight_timepoint_view",
    "five_day_view",
    "load_reduction_trajectories",
    "module_axis_sha256",
    "parse_reduction_scope",
]
