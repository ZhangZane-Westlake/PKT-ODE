"""Select broad and compound-specific responsive Genes before WGCNA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from .eligibility import FIT_MODEL_TIMES, MODEL_TIME_HOURS


SCHEMA_VERSION: Final[int] = 6
SELECTION_MODES: Final[set[str]] = {"organ", "single_compound"}


@dataclass(frozen=True)
class SelectionConfig:
    """Gene-selection thresholds.

    Args:
        broad_effect: Minimum absolute median log2FC for a broad response.
        strong_effect: Minimum absolute median log2FC for a specific response.
        broad_rate: Minimum fraction of fit conditions with a broad response.
        specific_rate: Minimum same-direction strong-condition fraction within one compound.
        noise_floor: Lower bound for pooled robust scale.
        chunk_size: Gene rows read per log2FC chunk.
    """

    broad_effect: float = 0.5
    strong_effect: float = 1.0
    broad_rate: float = 0.10
    specific_rate: float = 1.0 / 3.0
    noise_floor: float = 0.1
    chunk_size: int = 250

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any threshold is invalid.
        """

        if min(self.broad_effect, self.strong_effect, self.noise_floor) <= 0:
            raise ValueError("effect and noise thresholds must be positive")
        if self.strong_effect < self.broad_effect:
            raise ValueError("strong_effect must be at least broad_effect")
        if not 0 < self.broad_rate <= 1:
            raise ValueError("broad_rate must be in (0, 1]")
        if not 0 < self.specific_rate <= 1:
            raise ValueError("specific_rate must be in (0, 1]")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class SelectionResult:
    """Prepared selection artifacts.

    Args:
        gene_metrics: One row per input Gene.
        condition_metadata: One row per fit condition.
        fit_sample_metadata: One row per replicate used for reduction fitting.
        selected_gene_ids: Ordered selected Gene identifiers.
        selected_condition_medians: Condition-by-selected-Gene matrix.
        selected_fit_replicates: Replicate-by-selected-Gene log2FC matrix.
        directional_response_calls: Gene-by-condition directionality calls.
        broad_response_calls: Gene-by-condition broad response calls.
        strong_response_calls: Gene-by-condition strong response calls.
        metadata: Serializable selection contract.
    """

    gene_metrics: pd.DataFrame
    condition_metadata: pd.DataFrame
    fit_sample_metadata: pd.DataFrame
    selected_gene_ids: list[str]
    selected_condition_medians: np.ndarray
    selected_fit_replicates: np.ndarray
    directional_response_calls: np.ndarray
    broad_response_calls: np.ndarray
    strong_response_calls: np.ndarray
    metadata: dict[str, object]


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest.

    Args:
        path: File to digest.

    Returns:
        Hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_condition_layout(
    eligible_samples: pd.DataFrame,
    expected_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build deterministic fit-sample and condition axes.

    Args:
        eligible_samples: Eight-timepoint eligible sample metadata.
        expected_replicates: Required replicates per condition.

    Returns:
        Sorted fit samples, condition metadata, and condition-by-replicate indices.

    Raises:
        ValueError: If fit samples violate the contract.
    """

    required = {
        "sample_id",
        "condition_id",
        "treatment_id",
        "compound_no",
        "compound_name",
        "dose",
        "dose_unit",
        "model_time",
        "partition",
        "used_for_reduction_fit",
    }
    missing = sorted(required.difference(eligible_samples.columns))
    if missing:
        raise ValueError(f"eligible sample metadata is missing columns: {missing}")
    fit_flag = eligible_samples["used_for_reduction_fit"]
    if not pd.api.types.is_bool_dtype(fit_flag):
        fit_flag = fit_flag.astype(str).str.casefold().map({"true": True, "false": False})
    fit = eligible_samples.loc[fit_flag.eq(True)].copy()
    fit = fit[fit["partition"].astype(str).str.casefold().eq("train")]
    fit = fit[fit["model_time"].isin(FIT_MODEL_TIMES)]
    if fit.empty:
        raise ValueError("No training samples are marked for reduction fitting")
    fit["time_hours"] = fit["model_time"].map(MODEL_TIME_HOURS).astype(int)
    fit = fit.sort_values(["treatment_id", "time_hours", "sample_id"]).reset_index(drop=True)
    counts = fit.groupby("condition_id", observed=True).size()
    invalid = counts[counts.ne(expected_replicates)]
    if not invalid.empty:
        raise ValueError(f"fit conditions lost exact replicate coverage: {invalid.to_dict()}")
    condition_rows: list[dict[str, object]] = []
    replicate_indices: list[np.ndarray] = []
    for condition_id, group in fit.groupby("condition_id", sort=False, observed=True):
        first = group.iloc[0]
        condition_rows.append(
            {
                "condition_id": str(condition_id),
                "treatment_id": str(first["treatment_id"]),
                "compound_no": str(first["compound_no"]).zfill(5),
                "compound_name": str(first["compound_name"]),
                "dose": str(first["dose"]),
                "dose_unit": str(first["dose_unit"]),
                "model_time": str(first["model_time"]),
                "time_hours": int(first["time_hours"]),
                "n_replicates": len(group),
                "sample_ids": ";".join(group["sample_id"].astype(str)),
            }
        )
        replicate_indices.append(group.index.to_numpy(dtype=np.int64))
    conditions = pd.DataFrame(condition_rows)
    return fit, conditions, np.stack(replicate_indices, axis=0)


def _icc_one_way(values: np.ndarray) -> np.ndarray:
    """Calculate balanced one-way random-effects ICC(1,1).

    Args:
        values: Gene-by-condition-by-replicate values.

    Returns:
        Raw ICC values, which may be negative.
    """

    n_conditions = values.shape[1]
    n_replicates = values.shape[2]
    condition_means = values.mean(axis=2)
    grand_means = values.mean(axis=(1, 2))
    ms_between = n_replicates * np.sum(
        (condition_means - grand_means[:, None]) ** 2, axis=1
    ) / max(n_conditions - 1, 1)
    ms_within = np.sum(
        (values - condition_means[:, :, None]) ** 2, axis=(1, 2)
    ) / max(n_conditions * (n_replicates - 1), 1)
    denominator = ms_between + (n_replicates - 1) * ms_within
    return np.divide(
        ms_between - ms_within,
        denominator,
        out=np.full_like(ms_between, np.nan),
        where=denominator > 0,
    )


def _specific_support(
    strong_directions: np.ndarray,
    compounds: Sequence[str],
    required_rate: float,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray]:
    """Find Genes with repeated same-direction strong responses within a compound.

    Args:
        strong_directions: Gene-by-condition values in ``{-1, 0, 1}``.
        compounds: Compound identifier per condition.
        required_rate: Same-direction strong-condition fraction required within one compound.

    Returns:
        Specific flags, supporting compounds, signed support labels, maximum
        same-direction support counts, and maximum same-direction support rates.
    """

    compound_array = np.asarray(compounds, dtype=str)
    unique_compounds = sorted(set(compound_array.tolist()))
    positive_counts = np.zeros(
        (strong_directions.shape[0], len(unique_compounds)), dtype=np.int32
    )
    negative_counts = np.zeros_like(positive_counts)
    compound_condition_counts = np.zeros(len(unique_compounds), dtype=np.int32)
    for compound_index, compound in enumerate(unique_compounds):
        compound_directions = strong_directions[:, compound_array == compound]
        compound_condition_counts[compound_index] = compound_directions.shape[1]
        positive_counts[:, compound_index] = (compound_directions > 0).sum(axis=1)
        negative_counts[:, compound_index] = (compound_directions < 0).sum(axis=1)
    same_direction_counts = np.maximum(positive_counts, negative_counts)
    positive_rates = positive_counts / compound_condition_counts[None, :]
    negative_rates = negative_counts / compound_condition_counts[None, :]
    same_direction_rates = np.maximum(positive_rates, negative_rates)
    specific = (same_direction_rates >= required_rate).any(axis=1)
    supporting_compounds: list[str] = []
    supporting_directions: list[str] = []
    for gene_index in range(strong_directions.shape[0]):
        compound_labels: list[str] = []
        direction_labels: list[str] = []
        for compound_index, compound in enumerate(unique_compounds):
            positive = positive_rates[gene_index, compound_index] >= required_rate
            negative = negative_rates[gene_index, compound_index] >= required_rate
            if not positive and not negative:
                continue
            compound_labels.append(compound)
            direction = "+/-" if positive and negative else "+" if positive else "-"
            direction_labels.append(f"{compound}:{direction}")
        supporting_compounds.append(";".join(compound_labels))
        supporting_directions.append(";".join(direction_labels))
    return (
        specific,
        supporting_compounds,
        supporting_directions,
        same_direction_counts.max(axis=1),
        same_direction_rates.max(axis=1),
    )


def select_genes(
    log2fc_path: Path,
    eligible_samples: pd.DataFrame,
    config: SelectionConfig,
    expected_replicates: int = 3,
    selection_mode: str = "organ",
) -> SelectionResult:
    """Select responsive Genes from reduction-fit samples only.

    Args:
        log2fc_path: Wide Gene-by-sample log2FC matrix.
        eligible_samples: Eligible sample metadata from ``prepare``.
        config: Selection thresholds.
        expected_replicates: Exact replicate count.
        selection_mode: ``organ`` for Broad union Specific selection, or
            ``single_compound`` for Broad-only selection.

    Returns:
        Selection metrics, calls, selected condition medians, and selected
        replicate-level reduction-fit values.
    """

    config.validate()
    if selection_mode not in SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {sorted(SELECTION_MODES)}, got {selection_mode!r}"
        )
    if expected_replicates != 3:
        raise ValueError("Gene selection requires exactly three replicates")
    if not log2fc_path.is_file():
        raise FileNotFoundError(f"Missing log2FC matrix: {log2fc_path}")
    fit_samples, condition_metadata, replicate_indices = _build_condition_layout(
        eligible_samples, expected_replicates
    )
    sample_ids = fit_samples["sample_id"].astype(str).tolist()
    compounds = condition_metadata["compound_no"].astype(str).tolist()
    unique_compounds = sorted(set(compounds))
    if selection_mode == "single_compound" and len(unique_compounds) != 1:
        raise ValueError(
            "single_compound selection requires exactly one compound in fit samples, "
            f"found {unique_compounds}"
        )
    gene_rows: list[pd.DataFrame] = []
    selected_gene_ids: list[str] = []
    selected_medians: list[np.ndarray] = []
    selected_replicates: list[np.ndarray] = []
    directional_call_blocks: list[np.ndarray] = []
    broad_call_blocks: list[np.ndarray] = []
    strong_call_blocks: list[np.ndarray] = []
    seen: set[str] = set()

    try:
        reader = pd.read_csv(
            log2fc_path,
            sep="\t",
            usecols=["gene_id", *sample_ids],
            chunksize=config.chunk_size,
        )
        for chunk in reader:
            chunk["gene_id"] = chunk["gene_id"].astype(str)
            duplicated = chunk["gene_id"].isin(seen) | chunk["gene_id"].duplicated(False)
            if duplicated.any():
                duplicates = sorted(chunk.loc[duplicated, "gene_id"].unique())
                raise ValueError(f"log2FC matrix has duplicate Gene IDs: {duplicates[:10]}")
            seen.update(chunk["gene_id"])
            matrix = chunk[sample_ids].to_numpy(dtype=np.float64)
            values = matrix[:, replicate_indices]
            finite = np.isfinite(values).all(axis=(1, 2))
            variable = np.ptp(values, axis=(1, 2)) > 0
            qc_pass = finite & variable
            safe_values = np.where(np.isfinite(values), values, 0.0)
            medians = np.median(safe_values, axis=2)
            absolute_residuals = np.abs(safe_values - medians[:, :, None])
            pooled_mad = 1.4826 * np.median(absolute_residuals, axis=(1, 2))
            noise_scale = np.maximum(pooled_mad, config.noise_floor)
            median_signs = np.sign(medians).astype(np.int8)
            replicate_signs = np.sign(safe_values).astype(np.int8)
            directional_calls = (
                np.all(replicate_signs == median_signs[:, :, None], axis=2)
                & (median_signs != 0)
                & qc_pass[:, None]
            )
            broad_effect_calls = np.abs(medians) >= config.broad_effect
            broad_calls = (
                broad_effect_calls
                & directional_calls
                & qc_pass[:, None]
            )
            strong_calls = (
                (np.abs(medians) >= config.strong_effect)
                & directional_calls
                & qc_pass[:, None]
            )
            broad_rate = broad_calls.mean(axis=1)
            broad = broad_rate >= config.broad_rate
            strong_directions = np.where(strong_calls, median_signs, 0).astype(np.int8)
            if selection_mode == "single_compound":
                specific = np.zeros(len(chunk), dtype=bool)
                supporting_compounds = [""] * len(chunk)
                supporting_directions = [""] * len(chunk)
                max_specific_support = np.zeros(len(chunk), dtype=np.int32)
                max_specific_rate = np.zeros(len(chunk), dtype=np.float64)
            else:
                (
                    specific,
                    supporting_compounds,
                    supporting_directions,
                    max_specific_support,
                    max_specific_rate,
                ) = _specific_support(
                    strong_directions, compounds, config.specific_rate
                )
            selected = qc_pass & (broad | specific)
            classification = np.full(len(chunk), "excluded", dtype="<U8")
            classification[broad & ~specific] = "broad"
            classification[~broad & specific] = "specific"
            classification[broad & specific] = "both"

            same_sign = np.sum(
                np.sign(safe_values) == np.sign(medians)[:, :, None], axis=2
            )
            sign_agreement = same_sign.mean(axis=1) / expected_replicates
            within_sd = safe_values.std(axis=2, ddof=1).mean(axis=1)
            icc = _icc_one_way(safe_values)
            pair_means = (
                safe_values.sum(axis=2)[:, :, None] - safe_values
            ) / (expected_replicates - 1)
            loo_directional = np.zeros_like(pair_means, dtype=bool)
            for left_out in range(expected_replicates):
                retained = np.delete(safe_values, left_out, axis=2)
                pair_sign = np.sign(pair_means[:, :, left_out])
                loo_directional[:, :, left_out] = (
                    np.all(np.sign(retained) == pair_sign[:, :, None], axis=2)
                    & (pair_sign != 0)
                )
            loo_calls = (
                (np.abs(pair_means) >= config.broad_effect)
                & loo_directional
                & (np.sign(pair_means) == median_signs[:, :, None])
                & qc_pass[:, None, None]
            )
            broad_denominator = broad_calls.sum(axis=1) * expected_replicates
            loo_stable = np.divide(
                (loo_calls & broad_calls[:, :, None]).sum(axis=(1, 2)),
                broad_denominator,
                out=np.full(len(chunk), np.nan, dtype=np.float64),
                where=broad_denominator > 0,
            )
            exclusion_reason = np.where(
                ~finite,
                "non_finite_fit_values",
                np.where(~variable, "zero_fit_variance", np.where(~selected, "below_response_rules", "")),
            )
            metrics = pd.DataFrame(
                {
                    "gene_id": chunk["gene_id"].tolist(),
                    "qc_pass": qc_pass,
                    "selected": selected,
                    "selection_class": classification,
                    "directional_condition_rate": directional_calls.mean(axis=1),
                    "n_directional_conditions": directional_calls.sum(axis=1),
                    "broad_response_rate": broad_rate,
                    "n_broad_conditions": broad_calls.sum(axis=1),
                    "n_strong_conditions": strong_calls.sum(axis=1),
                    "max_abs_condition_median": np.max(np.abs(medians), axis=1),
                    "pooled_robust_mad": pooled_mad,
                    "noise_scale": noise_scale,
                    "max_robust_snr": np.max(np.abs(medians), axis=1) / noise_scale,
                    "mean_within_condition_sd": within_sd,
                    "icc_1_1": icc,
                    "mean_same_sign_fraction": sign_agreement,
                    "leave_one_out_broad_stability": loo_stable,
                    "max_strong_conditions_within_compound": max_specific_support,
                    "max_same_direction_strong_rate_within_compound": max_specific_rate,
                    "specific_supporting_compounds": supporting_compounds,
                    "specific_supporting_directions": supporting_directions,
                    "exclusion_reason": exclusion_reason,
                }
            )
            gene_rows.append(metrics)
            directional_call_blocks.append(directional_calls)
            broad_call_blocks.append(broad_calls)
            strong_call_blocks.append(strong_calls)
            if selected.any():
                selected_gene_ids.extend(chunk.loc[selected, "gene_id"].tolist())
                selected_medians.append(medians[selected])
                selected_replicates.append(matrix[selected])
    except ValueError as exc:
        if "Usecols do not match columns" in str(exc):
            raise ValueError("log2FC matrix is missing one or more fit sample columns") from exc
        raise
    if not seen:
        raise ValueError(f"log2FC matrix is empty: {log2fc_path}")
    if not selected_gene_ids:
        raise ValueError("Gene selection retained no Genes")

    gene_metrics = pd.concat(gene_rows, ignore_index=True)
    selected_matrix = np.concatenate(selected_medians, axis=0).T
    selected_replicate_matrix = np.concatenate(selected_replicates, axis=0).T
    directional_calls_all = np.concatenate(directional_call_blocks, axis=0)
    broad_calls_all = np.concatenate(broad_call_blocks, axis=0)
    strong_calls_all = np.concatenate(strong_call_blocks, axis=0)
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "selection_mode": selection_mode,
        "compound_numbers": unique_compounds,
        "fit_model_times": list(FIT_MODEL_TIMES),
        "expected_replicates": expected_replicates,
        "n_fit_samples": len(fit_samples),
        "n_fit_conditions": len(condition_metadata),
        "reduction_fit_unit": "replicate_log2fc",
        "n_input_genes": len(gene_metrics),
        "n_selected_genes": len(selected_gene_ids),
        "selection_counts": {
            str(key): int(value)
            for key, value in gene_metrics["selection_class"].value_counts().items()
        },
        "config": {
            "broad_effect": config.broad_effect,
            "strong_effect": config.strong_effect,
            "broad_rate": config.broad_rate,
            "specific_rate": config.specific_rate,
            "noise_floor": config.noise_floor,
            "chunk_size": config.chunk_size,
        },
        "selection_rules": {
            "directional": "all three replicate signs equal the non-zero condition-median sign",
            "broad": (
                "directional and abs(condition median) >= broad_effect in at least broad_rate of the single compound's fit conditions"
                if selection_mode == "single_compound"
                else "directional and abs(condition median) >= broad_effect in at least broad_rate of fit conditions"
            ),
            "specific": (
                "disabled"
                if selection_mode == "single_compound"
                else "same-sign directional conditions with abs(condition median) >= strong_effect occupy at least specific_rate of one compound's fit conditions"
            ),
            "robust_snr_role": "audit_only",
        },
        "inputs": {
            "log2fc": f"<input>/{log2fc_path.name}",
            "log2fc_sha256": sha256_file(log2fc_path),
        },
    }
    return SelectionResult(
        gene_metrics=gene_metrics,
        condition_metadata=condition_metadata,
        fit_sample_metadata=fit_samples,
        selected_gene_ids=selected_gene_ids,
        selected_condition_medians=selected_matrix,
        selected_fit_replicates=selected_replicate_matrix,
        directional_response_calls=directional_calls_all,
        broad_response_calls=broad_calls_all,
        strong_response_calls=strong_calls_all,
        metadata=metadata,
    )


def write_selection(result: SelectionResult, directory: Path) -> None:
    """Write Gene-selection artifacts without pickle.

    Args:
        result: Selection result.
        directory: Selection output directory.

    Raises:
        FileExistsError: If any target exists.
    """

    targets = {
        "metrics": directory / "gene_selection.tsv.gz",
        "selected": directory / "selected_genes.tsv",
        "conditions": directory / "condition_metadata.tsv",
        "condition_matrix_tsv": directory / "condition_medians.tsv.gz",
        "condition_matrix_npz": directory / "condition_medians.npz",
        "fit_samples": directory / "fit_sample_metadata.tsv",
        "fit_matrix_tsv": directory / "fit_replicate_values.tsv.gz",
        "fit_matrix_npz": directory / "fit_replicate_values.npz",
        "calls": directory / "response_calls.npz",
        "metadata": directory / "selection_manifest.json",
        "summary": directory / "summary.txt",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Selection outputs already exist: " + ", ".join(existing))
    directory.mkdir(parents=True, exist_ok=True)
    result.gene_metrics.to_csv(targets["metrics"], sep="\t", index=False)
    result.gene_metrics.loc[result.gene_metrics["selected"]].to_csv(
        targets["selected"], sep="\t", index=False
    )
    result.condition_metadata.to_csv(targets["conditions"], sep="\t", index=False)
    result.fit_sample_metadata.to_csv(targets["fit_samples"], sep="\t", index=False)
    matrix_frame = pd.DataFrame(
        result.selected_condition_medians,
        index=result.condition_metadata["condition_id"],
        columns=result.selected_gene_ids,
    )
    matrix_frame.index.name = "condition_id"
    matrix_frame.to_csv(
        targets["condition_matrix_tsv"], sep="\t", compression="gzip"
    )
    np.savez_compressed(
        targets["condition_matrix_npz"],
        values=result.selected_condition_medians.astype(np.float32),
        condition_ids=np.asarray(result.condition_metadata["condition_id"], dtype=str),
        gene_ids=np.asarray(result.selected_gene_ids, dtype=str),
    )
    fit_matrix_frame = pd.DataFrame(
        result.selected_fit_replicates,
        index=result.fit_sample_metadata["sample_id"],
        columns=result.selected_gene_ids,
    )
    fit_matrix_frame.index.name = "sample_id"
    fit_matrix_frame.to_csv(targets["fit_matrix_tsv"], sep="\t", compression="gzip")
    np.savez_compressed(
        targets["fit_matrix_npz"],
        values=result.selected_fit_replicates.astype(np.float32),
        sample_ids=np.asarray(result.fit_sample_metadata["sample_id"], dtype=str),
        gene_ids=np.asarray(result.selected_gene_ids, dtype=str),
    )
    np.savez_compressed(
        targets["calls"],
        directional=result.directional_response_calls.astype(np.uint8),
        broad=result.broad_response_calls.astype(np.uint8),
        strong=result.strong_response_calls.astype(np.uint8),
        gene_ids=np.asarray(result.gene_metrics["gene_id"], dtype=str),
        condition_ids=np.asarray(result.condition_metadata["condition_id"], dtype=str),
    )
    targets["metadata"].write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selection_counts = result.metadata["selection_counts"]
    config = result.metadata["config"]
    rules = result.metadata["selection_rules"]
    broad_rate_operator = ">="
    specific_rate_requirement = (
        "disabled"
        if result.metadata["selection_mode"] == "single_compound"
        else f">= {config['specific_rate']}"
    )
    summary_lines = [
        "Gene selection summary",
        "======================",
        f"selection_mode: {result.metadata['selection_mode']}",
        f"compound_numbers: {','.join(result.metadata['compound_numbers'])}",
        f"fit_model_times: {','.join(result.metadata['fit_model_times'])}",
        f"n_fit_conditions: {result.metadata['n_fit_conditions']}",
        f"n_fit_samples: {result.metadata['n_fit_samples']}",
        f"n_input_genes: {result.metadata['n_input_genes']}",
        f"n_selected_genes: {result.metadata['n_selected_genes']}",
        f"selection_counts: {json.dumps(selection_counts, sort_keys=True)}",
        f"broad_effect: {config['broad_effect']}",
        f"strong_effect: {config['strong_effect']}",
        f"broad_rate: {config['broad_rate']}",
        f"broad_rate_requirement: {broad_rate_operator} {config['broad_rate']}",
        f"specific_rate: {config['specific_rate']}",
        f"specific_rate_requirement: {specific_rate_requirement}",
        f"directional_rule: {rules['directional']}",
        f"broad_rule: {rules['broad']}",
        f"specific_rule: {rules['specific']}",
        "reduction_fit_unit: replicate_log2fc",
        "selected_gene_ids_file: selected_genes.tsv",
        "selection_manifest_file: selection_manifest.json",
    ]
    targets["summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
