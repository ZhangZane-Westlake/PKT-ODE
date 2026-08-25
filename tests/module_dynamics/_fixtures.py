"""Synthetic reduction-dir fixtures for module-dynamics tests."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


ELIGIBILITY_MODEL_TIMES: Final[tuple[str, ...]] = (
    "3H", "6H", "9H", "1D", "4D", "8D", "15D", "29D",
)
MODEL_TIME_HOURS: Final[dict[str, int]] = {
    "3H": 3, "6H": 6, "9H": 9, "1D": 24, "4D": 96,
    "8D": 192, "15D": 360, "29D": 696,
}


def write_synthetic_reduction(
    tmp_path: Path,
    n_compounds: int = 1,
    organ: str = "liver",
    compound_slug: str = "fenofibrate",
    module_count: int = 3,
    doses_per_compound: int = 2,
    seed: int = 7,
) -> Path:
    """Write a minimal but valid reduction config directory.

    Args:
        tmp_path: Temporary directory root.
        n_compounds: Number of distinct compounds.
        organ: Organ name.
        compound_slug: Compound directory slug (``None`` for organ scope).
        module_count: Number of non-grey modules.
        doses_per_compound: Treatments per compound.
        seed: Random seed for synthetic values.

    Returns:
        Path to the reduction config directory.
    """

    rng = np.random.default_rng(seed)
    base = (
        tmp_path
        / "output"
        / "gene_module_reduction"
        / "gene_selection_TEST"
        / organ
    )
    if compound_slug is None:
        scope_tail = base / "tight_stage" / "reductions" / "r2-0p8_test"
    else:
        scope_tail = (
            base / "compounds" / compound_slug / "tight_stage" / "reductions" / "r2-0p8_test"
        )
    projection = scope_tail / "projection"
    projection.mkdir(parents=True, exist_ok=True)

    gene_ids = [f"ENSRNOG{index:08d}" for index in range(6)]
    module_ids = [f"M{index:03d}" for index in range(1, module_count + 1)]
    loadings = np.zeros((module_count, len(gene_ids)), dtype=np.float64)
    for module_index in range(module_count):
        genes_for_module = slice(module_index * 2, module_index * 2 + 2)
        loadings[module_index, genes_for_module] = 1.0
    gene_center = rng.normal(size=len(gene_ids))
    gene_scale = np.full(len(gene_ids), 0.5)
    module_center = rng.normal(size=module_count)
    module_scale = np.full(module_count, 0.7)
    np.savez_compressed(
        projection / "module_bundle.npz",
        schema_version=np.asarray([2], dtype=np.int64),
        gene_ids=np.asarray(gene_ids, dtype=str),
        module_ids=np.asarray(module_ids, dtype=str),
        gene_center=gene_center,
        gene_scale=gene_scale,
        loadings=loadings,
        module_center=module_center,
        module_scale=module_scale,
    )

    compound_numbers = [f"{10001 + index:05d}" for index in range(n_compounds)]
    compound_names = [f"compound_{index}" for index in range(n_compounds)]
    doses = [10.0, 100.0][:doses_per_compound]
    treatment_meta_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    replicate_blocks: list[np.ndarray] = []
    for compound_index, compound_no in enumerate(compound_numbers):
        for dose in doses:
            treatment_id = f"ORGAN-{organ.upper()}__C{compound_no}__DOSE-{int(dose)}-MG-KG"
            treatment_meta_rows.append(
                {
                    "treatment_id": treatment_id,
                    "compound_no": compound_no,
                    "compound_name": compound_names[compound_index],
                    "dose": str(int(dose)),
                    "dose_unit": "MG-KG",
                    "organ": organ.capitalize(),
                }
            )
            per_time: list[np.ndarray] = []
            for model_time in ELIGIBILITY_MODEL_TIMES:
                base_value = rng.normal(scale=0.4, size=module_count)
                replicate_values = base_value[None, :] + rng.normal(
                    scale=0.1, size=(3, module_count)
                )
                per_time.append(replicate_values)
                regimen = "single" if model_time in ("3H", "6H", "9H", "1D") else "repeat"
                for rep_index in range(3):
                    sample_id = f"{treatment_id}__{model_time}_R{rep_index}"
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "condition_id": f"{treatment_id}__TIME-{model_time}",
                            "treatment_id": treatment_id,
                            "compound_no": compound_no,
                            "compound_name": compound_names[compound_index],
                            "dose": str(int(dose)),
                            "dose_unit": "MG-KG",
                            "dose_level": "low" if dose < 50 else "high",
                            "time_label": model_time,
                            "model_time": model_time,
                            "time_hours": MODEL_TIME_HOURS[model_time],
                            "regimen": regimen,
                            "organ": organ.capitalize(),
                            "partition": "train",
                            "used_for_reduction_fit": model_time
                            in ("3H", "6H", "9H", "1D", "4D", "8D"),
                        }
                    )
            replicate_blocks.append(np.stack(per_time, axis=0))  # [8, 3, M]
    replicate_values = np.stack(replicate_blocks, axis=0).astype(np.float32)  # [T,8,3,M]
    mean_values = replicate_values.mean(axis=2).astype(np.float32)
    sd_values = replicate_values.std(axis=2, ddof=1).astype(np.float32)
    treatment_ids = [row["treatment_id"] for row in treatment_meta_rows]
    np.savez_compressed(
        projection / "treatment_module_trajectories.npz",
        replicate_values=replicate_values,
        mean_values=mean_values,
        sd_values=sd_values,
        treatment_ids=np.asarray(treatment_ids, dtype=str),
        model_times=np.asarray(ELIGIBILITY_MODEL_TIMES, dtype=str),
        module_ids=np.asarray(module_ids, dtype=str),
    )
    pd.DataFrame(sample_rows).to_csv(
        projection / "sample_metadata.tsv.gz", sep="\t", index=False
    )
    return scope_tail


__all__ = ["write_synthetic_reduction"]
