"""Fit and apply fixed Gene-to-WGCNA-eigengene projections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd

from .eligibility import ELIGIBILITY_MODEL_TIMES, MODEL_TIME_HOURS
from .selection import sha256_file


BUNDLE_SCHEMA_VERSION: Final[int] = 2


@dataclass(frozen=True)
class ModuleBundle:
    """Fixed projection parameters.

    Args:
        gene_ids: Ordered selected Gene axis.
        module_ids: Ordered non-grey module axis.
        gene_center: Training mean per Gene.
        gene_scale: Training standard deviation per Gene.
        loadings: Module-by-Gene fixed PC1 loadings.
        module_center: Training PC1 score mean per module.
        module_scale: Training PC1 score standard deviation per module.
    """

    gene_ids: list[str]
    module_ids: list[str]
    gene_center: np.ndarray
    gene_scale: np.ndarray
    loadings: np.ndarray
    module_center: np.ndarray
    module_scale: np.ndarray


@dataclass(frozen=True)
class ProjectionFitResult:
    """Fixed projection fit artifacts.

    Args:
        bundle: Fixed projection parameters.
        fit_sample_scores_raw: Training-replicate PC1 scores.
        fit_sample_scores_standardized: Standardized training-replicate module scores.
        module_summary: Per-module projection audit.
        gene_modules: Ordered Gene-to-module membership table.
        manifest: Serializable projection contract.
    """

    bundle: ModuleBundle
    fit_sample_scores_raw: np.ndarray
    fit_sample_scores_standardized: np.ndarray
    module_summary: pd.DataFrame
    gene_modules: pd.DataFrame
    manifest: dict[str, object]


@dataclass(frozen=True)
class TransformedModules:
    """Projected module values for samples and complete trajectories.

    Args:
        sample_values_raw: Sample-by-module raw PC1 scores.
        sample_values_standardized: Sample-by-module standardized scores.
        sample_metadata: Ordered sample metadata.
        treatment_ids: Ordered trajectory treatment IDs.
        replicate_values: Treatment-by-time-by-replicate-by-module scores.
        mean_values: Treatment-by-time-by-module means.
        sd_values: Treatment-by-time-by-module sample standard deviations.
    """

    sample_values_raw: np.ndarray
    sample_values_standardized: np.ndarray
    sample_metadata: pd.DataFrame
    treatment_ids: list[str]
    replicate_values: np.ndarray
    mean_values: np.ndarray
    sd_values: np.ndarray


def _axis_digest(values: list[str]) -> str:
    """Digest an ordered string axis.

    Args:
        values: Ordered axis labels.

    Returns:
        SHA-256 hexadecimal digest.
    """

    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _finite_matmul(left: np.ndarray, right: np.ndarray, context: str) -> np.ndarray:
    """Multiply arrays and reject any non-finite output.

    Args:
        left: Left matrix operand.
        right: Right matrix operand.
        context: Human-readable operation label for errors.

    Returns:
        Finite matrix product.
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right
    if not np.isfinite(result).all():
        raise ValueError(f"{context} produced non-finite values")
    return result


def _read_fit_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Read the no-pickle replicate-level reduction-fit cache.

    Args:
        path: Replicate-by-selected-Gene NPZ.

    Returns:
        Values, sample IDs, and Gene IDs.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Missing reduction-fit matrix: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float64, copy=False)
        sample_ids = archive["sample_ids"].astype(str).tolist()
        gene_ids = archive["gene_ids"].astype(str).tolist()
    if values.shape != (len(sample_ids), len(gene_ids)):
        raise ValueError(f"Invalid replicate-fit shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("replicate-fit matrix contains non-finite values")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("replicate-fit sample axis contains duplicates")
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("replicate-fit Gene axis contains duplicates")
    return values, sample_ids, gene_ids


def fit_fixed_projection(
    fit_matrix_path: Path,
    gene_modules: pd.DataFrame,
    gene_metrics: Optional[pd.DataFrame] = None,
) -> ProjectionFitResult:
    """Fit fixed train-only PC1 loadings for WGCNA modules.

    Args:
        fit_matrix_path: Training-replicate-by-selected-Gene NPZ.
        gene_modules: WGCNA Gene membership with ``gene_id`` and ``module_id``.
        gene_metrics: Optional Gene-selection metrics for module composition audits.

    Returns:
        Projection bundle, scores, and module summaries.
    """

    required = {"gene_id", "module_id"}
    missing = sorted(required.difference(gene_modules.columns))
    if missing:
        raise ValueError(f"WGCNA membership is missing columns: {missing}")
    values, sample_ids, gene_ids = _read_fit_matrix(fit_matrix_path)
    membership = gene_modules.copy()
    membership["gene_id"] = membership["gene_id"].astype(str)
    membership["module_id"] = membership["module_id"].fillna("grey").astype(str)
    if membership["gene_id"].duplicated(keep=False).any():
        raise ValueError("WGCNA membership contains duplicate Gene IDs")
    if set(membership["gene_id"]) != set(gene_ids):
        missing_membership = sorted(set(gene_ids).difference(membership["gene_id"]))
        extra_membership = sorted(set(membership["gene_id"]).difference(gene_ids))
        raise ValueError(
            "WGCNA and selection Gene axes differ: "
            f"missing={missing_membership[:5]}, extra={extra_membership[:5]}"
        )
    membership = membership.set_index("gene_id").loc[gene_ids].reset_index()
    module_ids = sorted(
        value for value in membership["module_id"].unique() if value.casefold() != "grey"
    )
    if not module_ids:
        raise ValueError("WGCNA produced no non-grey modules")

    gene_center = values.mean(axis=0)
    gene_scale = values.std(axis=0, ddof=1)
    if not np.isfinite(gene_scale).all() or np.any(gene_scale <= 0):
        bad = [gene_ids[index] for index in np.flatnonzero(gene_scale <= 0)[:10]]
        raise ValueError(f"Selected Genes have invalid training scale: {bad}")
    standardized = (values - gene_center) / gene_scale
    loadings = np.zeros((len(module_ids), len(gene_ids)), dtype=np.float64)
    raw_scores = np.zeros((len(sample_ids), len(module_ids)), dtype=np.float64)
    summary_rows: list[dict[str, object]] = []
    gene_index = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    selection_lookup: Optional[pd.DataFrame] = None
    if gene_metrics is not None:
        if not {"gene_id", "selection_class"}.issubset(gene_metrics.columns):
            raise ValueError("Gene metrics must contain gene_id and selection_class")
        selection_lookup = gene_metrics.set_index(gene_metrics["gene_id"].astype(str))

    for module_index, module_id in enumerate(module_ids):
        module_genes = membership.loc[membership["module_id"].eq(module_id), "gene_id"].tolist()
        indices = np.asarray([gene_index[gene_id] for gene_id in module_genes], dtype=np.int64)
        module_values = standardized[:, indices]
        _, singular_values, right_vectors = np.linalg.svd(module_values, full_matrices=False)
        loading = right_vectors[0]
        score = _finite_matmul(module_values, loading, f"fixed projection for {module_id}")
        mean_expression = module_values.mean(axis=1)
        correlation = np.corrcoef(score, mean_expression)[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            loading = -loading
            score = -score
            correlation = -correlation
        loadings[module_index, indices] = loading
        raw_scores[:, module_index] = score
        total_variance = float(np.sum(singular_values**2))
        pve = float(singular_values[0] ** 2 / total_variance) if total_variance > 0 else 0.0
        class_counts: dict[str, int] = {}
        if selection_lookup is not None:
            class_counts = {
                str(key): int(value)
                for key, value in selection_lookup.loc[module_genes, "selection_class"]
                .value_counts()
                .items()
            }
        summary_rows.append(
            {
                "module_id": module_id,
                "n_genes": len(module_genes),
                "pc1_variance_explained": pve,
                "pc1_mean_expression_correlation": float(correlation),
                "n_broad_genes": class_counts.get("broad", 0),
                "n_specific_genes": class_counts.get("specific", 0),
                "n_both_genes": class_counts.get("both", 0),
                "initial_status": "primary" if pve >= 0.3 else "exploratory",
                "initial_status_reason": "" if pve >= 0.3 else "pc1_variance_explained_lt_0.3",
            }
        )

    module_center = raw_scores.mean(axis=0)
    module_scale = raw_scores.std(axis=0, ddof=1)
    if not np.isfinite(module_scale).all() or np.any(module_scale <= 0):
        bad = [module_ids[index] for index in np.flatnonzero(module_scale <= 0)]
        raise ValueError(f"Modules have invalid training score scale: {bad}")
    standardized_scores = (raw_scores - module_center) / module_scale
    bundle = ModuleBundle(
        gene_ids=gene_ids,
        module_ids=module_ids,
        gene_center=gene_center,
        gene_scale=gene_scale,
        loadings=loadings,
        module_center=module_center,
        module_scale=module_scale,
    )
    manifest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "n_fit_samples": len(sample_ids),
        "fit_unit": "replicate_log2fc",
        "n_selected_genes": len(gene_ids),
        "n_modules": len(module_ids),
        "gene_axis_sha256": _axis_digest(gene_ids),
        "module_axis_sha256": _axis_digest(module_ids),
        "orientation_rule": "positive correlation with mean standardized module expression",
        "module_score_definition": "train-standardized fixed module PC1 score",
        "inputs": {
            "fit_matrix": f"<selection>/{fit_matrix_path.name}",
            "fit_matrix_sha256": sha256_file(fit_matrix_path),
        },
    }
    return ProjectionFitResult(
        bundle=bundle,
        fit_sample_scores_raw=raw_scores,
        fit_sample_scores_standardized=standardized_scores,
        module_summary=pd.DataFrame(summary_rows),
        gene_modules=membership,
        manifest=manifest,
    )


def write_projection_fit(
    result: ProjectionFitResult,
    sample_ids: list[str],
    directory: Path,
) -> None:
    """Write a fixed projection fit without overwriting artifacts.

    Args:
        result: Projection fit result.
        sample_ids: Ordered training replicate sample IDs.
        directory: Projection output directory.
    """

    targets = {
        "bundle": directory / "module_bundle.npz",
        "scores": directory / "fit_sample_module_values.npz",
        "summary": directory / "module_projection_summary.tsv",
        "membership": directory / "gene_modules.tsv.gz",
        "manifest": directory / "projection_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Projection outputs already exist: " + ", ".join(existing))
    if result.fit_sample_scores_raw.shape[0] != len(sample_ids):
        raise ValueError("sample ID axis does not match projection scores")
    directory.mkdir(parents=True, exist_ok=True)
    bundle = result.bundle
    np.savez_compressed(
        targets["bundle"],
        schema_version=np.asarray([BUNDLE_SCHEMA_VERSION], dtype=np.int64),
        gene_ids=np.asarray(bundle.gene_ids, dtype=str),
        module_ids=np.asarray(bundle.module_ids, dtype=str),
        gene_center=bundle.gene_center,
        gene_scale=bundle.gene_scale,
        loadings=bundle.loadings,
        module_center=bundle.module_center,
        module_scale=bundle.module_scale,
    )
    np.savez_compressed(
        targets["scores"],
        raw_values=result.fit_sample_scores_raw.astype(np.float32),
        standardized_values=result.fit_sample_scores_standardized.astype(np.float32),
        sample_ids=np.asarray(sample_ids, dtype=str),
        module_ids=np.asarray(bundle.module_ids, dtype=str),
    )
    result.module_summary.to_csv(targets["summary"], sep="\t", index=False)
    result.gene_modules.to_csv(targets["membership"], sep="\t", index=False)
    targets["manifest"].write_text(
        json.dumps(result.manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_module_bundle(path: Path) -> ModuleBundle:
    """Load and validate a fixed module bundle.

    Args:
        path: Bundle NPZ path.

    Returns:
        Validated module bundle.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Missing module bundle: {path}")
    with np.load(path, allow_pickle=False) as archive:
        schema_version = int(archive["schema_version"][0])
        gene_ids = archive["gene_ids"].astype(str).tolist()
        module_ids = archive["module_ids"].astype(str).tolist()
        gene_center = archive["gene_center"].astype(np.float64, copy=False)
        gene_scale = archive["gene_scale"].astype(np.float64, copy=False)
        loadings = archive["loadings"].astype(np.float64, copy=False)
        module_center = archive["module_center"].astype(np.float64, copy=False)
        module_scale = archive["module_scale"].astype(np.float64, copy=False)
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported module bundle schema: {schema_version}")
    if len(set(gene_ids)) != len(gene_ids) or len(set(module_ids)) != len(module_ids):
        raise ValueError("module bundle axes contain duplicates")
    if loadings.shape != (len(module_ids), len(gene_ids)):
        raise ValueError(f"Invalid loading shape: {loadings.shape}")
    for name, values in {
        "gene_center": gene_center,
        "gene_scale": gene_scale,
        "loadings": loadings,
        "module_center": module_center,
        "module_scale": module_scale,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"module bundle {name} contains non-finite values")
    if np.any(gene_scale <= 0) or np.any(module_scale <= 0):
        raise ValueError("module bundle contains non-positive scales")
    return ModuleBundle(
        gene_ids=gene_ids,
        module_ids=module_ids,
        gene_center=gene_center,
        gene_scale=gene_scale,
        loadings=loadings,
        module_center=module_center,
        module_scale=module_scale,
    )


def project_values(values: np.ndarray, bundle: ModuleBundle) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed bundle to sample-by-Gene values.

    Args:
        values: Sample-by-Gene matrix in bundle Gene order.
        bundle: Fixed train-only module bundle.

    Returns:
        Raw and train-standardized sample-by-module values.
    """

    if values.shape[-1] != len(bundle.gene_ids):
        raise ValueError("input Gene axis does not match module bundle")
    if not np.isfinite(values).all():
        raise ValueError("input Gene values contain non-finite values")
    standardized_genes = (values - bundle.gene_center) / bundle.gene_scale
    raw = _finite_matmul(
        standardized_genes,
        bundle.loadings.T,
        "fixed module projection",
    )
    standardized = (raw - bundle.module_center) / bundle.module_scale
    return raw, standardized


def read_selected_gene_values(
    log2fc_path: Path,
    sample_ids: list[str],
    gene_ids: list[str],
    chunk_size: int = 250,
) -> np.ndarray:
    """Read selected Genes and requested samples from a wide log2FC matrix.

    Args:
        log2fc_path: Wide Gene-by-sample matrix.
        sample_ids: Ordered requested sample IDs.
        gene_ids: Ordered bundle Gene IDs.
        chunk_size: Input Gene rows per chunk.

    Returns:
        Sample-by-Gene matrix in bundle order.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    requested = set(gene_ids)
    found: dict[str, np.ndarray] = {}
    try:
        reader = pd.read_csv(
            log2fc_path,
            sep="\t",
            usecols=["gene_id", *sample_ids],
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk["gene_id"] = chunk["gene_id"].astype(str)
            selected = chunk[chunk["gene_id"].isin(requested)]
            selected_values = selected.loc[:, sample_ids].to_numpy(dtype=np.float64)
            for gene_id, row_values in zip(
                selected["gene_id"].astype(str), selected_values
            ):
                if gene_id in found:
                    raise ValueError(f"log2FC matrix has duplicate Gene ID: {gene_id}")
                found[gene_id] = row_values
    except ValueError as exc:
        if "Usecols do not match columns" in str(exc):
            raise ValueError("log2FC matrix is missing one or more transform samples") from exc
        raise
    missing = [gene_id for gene_id in gene_ids if gene_id not in found]
    if missing:
        raise ValueError(f"log2FC matrix is missing bundle Genes: {missing[:10]}")
    gene_by_sample = np.stack([found[gene_id] for gene_id in gene_ids], axis=0)
    return gene_by_sample.T


def transform_samples(
    log2fc_path: Path,
    sample_metadata: pd.DataFrame,
    bundle: ModuleBundle,
    expected_replicates: int = 3,
    chunk_size: int = 250,
) -> TransformedModules:
    """Project eligible samples and assemble complete module trajectories.

    Args:
        log2fc_path: Wide Gene-by-sample matrix.
        sample_metadata: Eligible sample metadata.
        bundle: Fixed projection bundle.
        expected_replicates: Exact replicates required per trajectory cell.
        chunk_size: Input Gene rows per chunk.

    Returns:
        Sample and treatment module values.
    """

    required = {"sample_id", "treatment_id", "model_time"}
    missing = sorted(required.difference(sample_metadata.columns))
    if missing:
        raise ValueError(f"transform metadata is missing columns: {missing}")
    metadata = sample_metadata.copy()
    metadata["time_hours"] = metadata["model_time"].map(MODEL_TIME_HOURS)
    if metadata["time_hours"].isna().any():
        raise ValueError("transform metadata contains unsupported model times")
    metadata = metadata.sort_values(
        ["partition", "treatment_id", "time_hours", "sample_id"]
        if "partition" in metadata.columns
        else ["treatment_id", "time_hours", "sample_id"]
    ).reset_index(drop=True)
    sample_ids = metadata["sample_id"].astype(str).tolist()
    gene_values = read_selected_gene_values(
        log2fc_path, sample_ids, bundle.gene_ids, chunk_size=chunk_size
    )
    raw, standardized = project_values(gene_values, bundle)
    treatment_ids = sorted(metadata["treatment_id"].astype(str).unique())
    replicate_values = np.empty(
        (
            len(treatment_ids),
            len(ELIGIBILITY_MODEL_TIMES),
            expected_replicates,
            len(bundle.module_ids),
        ),
        dtype=np.float64,
    )
    for treatment_index, treatment_id in enumerate(treatment_ids):
        treatment_mask = metadata["treatment_id"].astype(str).eq(treatment_id)
        for time_index, model_time in enumerate(ELIGIBILITY_MODEL_TIMES):
            indices = np.flatnonzero(
                (treatment_mask & metadata["model_time"].eq(model_time)).to_numpy()
            )
            if len(indices) != expected_replicates:
                raise ValueError(
                    f"trajectory cell {treatment_id}/{model_time} has {len(indices)} replicates"
                )
            replicate_values[treatment_index, time_index] = standardized[indices]
    mean_values = replicate_values.mean(axis=2)
    sd_values = replicate_values.std(axis=2, ddof=1)
    return TransformedModules(
        sample_values_raw=raw,
        sample_values_standardized=standardized,
        sample_metadata=metadata,
        treatment_ids=treatment_ids,
        replicate_values=replicate_values,
        mean_values=mean_values,
        sd_values=sd_values,
    )


def write_transformed_modules(
    transformed: TransformedModules,
    bundle: ModuleBundle,
    directory: Path,
) -> None:
    """Write projected sample and trajectory artifacts.

    Args:
        transformed: Transformed values.
        bundle: Fixed module bundle.
        directory: Projection output directory.
    """

    targets = {
        "samples": directory / "sample_module_values.npz",
        "metadata": directory / "sample_metadata.tsv.gz",
        "trajectories": directory / "treatment_module_trajectories.npz",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Transformed outputs already exist: " + ", ".join(existing))
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        targets["samples"],
        raw_values=transformed.sample_values_raw.astype(np.float32),
        standardized_values=transformed.sample_values_standardized.astype(np.float32),
        sample_ids=np.asarray(transformed.sample_metadata["sample_id"], dtype=str),
        module_ids=np.asarray(bundle.module_ids, dtype=str),
    )
    transformed.sample_metadata.to_csv(targets["metadata"], sep="\t", index=False)
    np.savez_compressed(
        targets["trajectories"],
        replicate_values=transformed.replicate_values.astype(np.float32),
        mean_values=transformed.mean_values.astype(np.float32),
        sd_values=transformed.sd_values.astype(np.float32),
        treatment_ids=np.asarray(transformed.treatment_ids, dtype=str),
        model_times=np.asarray(ELIGIBILITY_MODEL_TIMES, dtype=str),
        module_ids=np.asarray(bundle.module_ids, dtype=str),
    )
