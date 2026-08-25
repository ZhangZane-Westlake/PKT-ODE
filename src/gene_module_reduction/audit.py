"""Audit fixed WGCNA modules for stability and Hallmark purity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .eligibility import EARLY_MODEL_TIMES
from .projection import ModuleBundle, load_module_bundle
from .selection import sha256_file


REQUIRED_GENE_SET_SOURCES: Final[set[str]] = {"hallmark"}


@dataclass(frozen=True)
class GeneSetCollection:
    """Versioned long-form Gene sets.

    Args:
        genes: Long table with source, term, and Gene identifiers.
        manifest: Verified source metadata.
    """

    genes: pd.DataFrame
    manifest: pd.DataFrame


def load_gene_sets(manifest_path: Path, allow_partial: bool = False) -> GeneSetCollection:
    """Load versioned Gene-set snapshots and verify their hashes.

    Args:
        manifest_path: TSV with source, version, path, and sha256.
        allow_partial: Retained for compatibility with earlier callers.

    Returns:
        Normalized long Gene-set collection.

    Raises:
        ValueError: If source schemas, versions, or hashes are invalid.
    """

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Gene-set manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source", "version", "path", "sha256"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Gene-set manifest is missing columns: {missing}")
    manifest = manifest[list(required)].copy()
    manifest["source"] = manifest["source"].str.strip().str.casefold()
    manifest["version"] = manifest["version"].str.strip()
    if manifest["source"].duplicated(keep=False).any():
        raise ValueError("Gene-set manifest has duplicate sources")
    if (manifest["version"] == "").any() or (manifest["sha256"] == "").any():
        raise ValueError("Every Gene-set source needs a version and SHA-256")
    sources = set(manifest["source"])
    missing_sources = sorted(REQUIRED_GENE_SET_SOURCES.difference(sources))
    if missing_sources and not allow_partial:
        raise ValueError(f"Gene-set manifest is missing required sources: {missing_sources}")

    normalized_blocks: list[pd.DataFrame] = []
    resolved_paths: list[str] = []
    for row in manifest.itertuples(index=False):
        source_path = Path(str(row.path))
        if not source_path.is_absolute():
            source_path = (manifest_path.parent / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing {row.source} Gene-set snapshot: {source_path}")
        actual_sha256 = sha256_file(source_path)
        if actual_sha256.casefold() != str(row.sha256).casefold():
            raise ValueError(
                f"{row.source} SHA-256 mismatch: expected {row.sha256}, got {actual_sha256}"
            )
        table = pd.read_csv(source_path, sep=None, engine="python", dtype=str)
        if {"term_id", "term_name", "gene_id"}.issubset(table.columns):
            normalized = table[["term_id", "term_name", "gene_id"]].copy()
        elif {"gs_name", "ensembl_gene"}.issubset(table.columns):
            normalized = table[["gs_name", "ensembl_gene"]].rename(
                columns={"gs_name": "term_name", "ensembl_gene": "gene_id"}
            )
            normalized["term_id"] = normalized["term_name"]
        else:
            raise ValueError(
                f"{row.source} snapshot must contain term_id/term_name/gene_id "
                "or gs_name/ensembl_gene"
            )
        normalized["source"] = str(row.source)
        normalized["version"] = str(row.version)
        normalized_blocks.append(normalized)
        resolved_paths.append(str(source_path))
    manifest["resolved_path"] = resolved_paths
    genes = pd.concat(normalized_blocks, ignore_index=True)
    genes = genes.dropna(subset=["term_id", "term_name", "gene_id"])
    genes = genes.astype(str).drop_duplicates(
        ["source", "version", "term_id", "gene_id"]
    )
    return GeneSetCollection(genes=genes, manifest=manifest)


def calculate_module_hallmark_purity(
    gene_modules: pd.DataFrame,
    gene_sets: GeneSetCollection,
) -> pd.DataFrame:
    """Calculate a complete module-by-Hallmark purity matrix.

    Args:
        gene_modules: Selected Gene-to-module assignments including grey Genes.
        gene_sets: Versioned Gene-set collection.

    Returns:
        Rows are non-grey modules, columns are Hallmark names, and each value is
        ``module-Hallmark overlap / module Gene count``.

    Raises:
        ValueError: If inputs are invalid or Hallmark names are ambiguous.
    """

    required = {"gene_id", "module_id"}
    missing = sorted(required.difference(gene_modules.columns))
    if missing:
        raise ValueError(f"Gene modules are missing columns: {missing}")
    hallmark = gene_sets.genes[
        gene_sets.genes["source"].astype(str).str.casefold().eq("hallmark")
    ].copy()
    if hallmark.empty:
        raise ValueError("Gene-set collection does not contain Hallmark")
    term_names_per_id = hallmark.groupby("term_id", observed=True)["term_name"].nunique()
    if term_names_per_id.gt(1).any():
        ambiguous = term_names_per_id[term_names_per_id.gt(1)].index.astype(str).tolist()
        raise ValueError(f"Hallmark term IDs have multiple names: {ambiguous[:5]}")
    term_ids_per_name = hallmark.groupby("term_name", observed=True)["term_id"].nunique()
    if term_ids_per_name.gt(1).any():
        ambiguous = term_ids_per_name[term_ids_per_name.gt(1)].index.astype(str).tolist()
        raise ValueError(f"Hallmark names have multiple term IDs: {ambiguous[:5]}")
    module_genes = {
        str(module_id): set(group["gene_id"].astype(str))
        for module_id, group in gene_modules.groupby("module_id", observed=True)
        if str(module_id).casefold() != "grey"
    }
    if not module_genes:
        raise ValueError("Gene modules contain no non-grey modules")
    hallmark_terms = {
        str(term_name): set(term_table["gene_id"].astype(str))
        for term_name, term_table in hallmark.groupby("term_name", observed=True)
    }
    purity = pd.DataFrame(
        index=sorted(module_genes),
        columns=sorted(hallmark_terms),
        dtype=float,
    )
    purity.index.name = "module_id"
    for module_id, members in module_genes.items():
        for term_name, term_genes in hallmark_terms.items():
            purity.loc[module_id, term_name] = (
                len(members.intersection(term_genes)) / len(members)
            )
    return purity


def _refit_loading(values: np.ndarray, reference: np.ndarray) -> float:
    """Refit PC1 and return absolute cosine with a reference loading.

    Args:
        values: Subsample replicate-by-module-Gene values.
        reference: Full-data oriented PC1 loading.

    Returns:
        Absolute cosine similarity.
    """

    if values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan")
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    loading = right_vectors[0]
    denominator = np.linalg.norm(loading) * np.linalg.norm(reference)
    if denominator <= 0:
        return float("nan")
    return float(abs(np.dot(loading, reference) / denominator))


def calculate_module_stability(
    fit_matrix_path: Path,
    fit_sample_metadata: pd.DataFrame,
    bundle: ModuleBundle,
    n_subsamples: int = 20,
    compound_fraction: float = 0.8,
    recovery_cosine: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Audit fixed-membership eigengene stability across compound subsets and phases.

    Args:
        fit_matrix_path: Training-replicate-by-selected-Gene NPZ.
        fit_sample_metadata: Ordered training replicate metadata.
        bundle: Fixed full-data module bundle.
        n_subsamples: Number of compound subsamples.
        compound_fraction: Fraction of compounds per subsample.
        recovery_cosine: Cosine threshold counted as recovered.
        seed: Deterministic sampling seed.

    Returns:
        Per-module stability metrics.
    """

    if n_subsamples <= 0 or not 0 < compound_fraction <= 1:
        raise ValueError("invalid stability subsampling configuration")
    with np.load(fit_matrix_path, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float64, copy=False)
        sample_ids = archive["sample_ids"].astype(str).tolist()
        gene_ids = archive["gene_ids"].astype(str).tolist()
    if gene_ids != bundle.gene_ids:
        raise ValueError("stability and module bundle Gene axes differ")
    if fit_sample_metadata["sample_id"].astype(str).tolist() != sample_ids:
        raise ValueError("stability sample metadata order differs from replicate-fit matrix")
    standardized = (values - bundle.gene_center) / bundle.gene_scale
    compounds = fit_sample_metadata["compound_no"].astype(str).to_numpy()
    unique_compounds = np.asarray(sorted(set(compounds.tolist())), dtype=str)
    subset_size = min(
        len(unique_compounds),
        max(2, int(math.ceil(len(unique_compounds) * compound_fraction))),
    )
    rng = np.random.default_rng(seed)
    subset_masks = [
        np.isin(compounds, rng.choice(unique_compounds, size=subset_size, replace=False))
        for _ in range(n_subsamples)
    ]
    early_mask = fit_sample_metadata["model_time"].isin(EARLY_MODEL_TIMES).to_numpy()
    repeat_mask = fit_sample_metadata["model_time"].isin({"4D", "8D"}).to_numpy()
    rows: list[dict[str, object]] = []
    for module_index, module_id in enumerate(bundle.module_ids):
        gene_indices = np.flatnonzero(np.abs(bundle.loadings[module_index]) > 0)
        reference = bundle.loadings[module_index, gene_indices]
        cosines = np.asarray(
            [
                _refit_loading(standardized[mask][:, gene_indices], reference)
                for mask in subset_masks
            ],
            dtype=np.float64,
        )
        early_cosine = _refit_loading(standardized[early_mask][:, gene_indices], reference)
        repeat_cosine = _refit_loading(standardized[repeat_mask][:, gene_indices], reference)
        rows.append(
            {
                "module_id": module_id,
                "n_genes": len(gene_indices),
                "n_compound_subsamples": n_subsamples,
                "compound_subsample_fraction": compound_fraction,
                "median_subsample_loading_cosine": float(np.nanmedian(cosines)),
                "subsample_recovery_rate": float(np.nanmean(cosines >= recovery_cosine)),
                "early_loading_cosine": early_cosine,
                "repeat_loading_cosine": repeat_cosine,
                "early_repeat_preservation": float(np.nanmin([early_cosine, repeat_cosine])),
            }
        )
    return pd.DataFrame(rows)


def write_audit(
    module_summary: pd.DataFrame,
    stability: pd.DataFrame,
    module_hallmark_purity: pd.DataFrame,
    gene_sets: GeneSetCollection,
    directory: Path,
) -> pd.DataFrame:
    """Write combined module audit artifacts.

    Args:
        module_summary: Projection-level module metrics.
        stability: Compound and phase stability metrics.
        module_hallmark_purity: Module-by-Hallmark purity matrix.
        gene_sets: Verified Gene-set collection.
        directory: Audit output directory.

    Returns:
        Combined module audit table.
    """

    targets = {
        "stability": directory / "module_stability.tsv",
        "hallmark_purity": directory / "module_hallmark_purity.tsv",
        "combined": directory / "module_audit.tsv",
        "sources": directory / "gene_set_manifest_used.tsv",
        "summary": directory / "summary.txt",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Audit outputs already exist: " + ", ".join(existing))
    combined = module_summary.merge(stability, on=["module_id", "n_genes"], validate="one_to_one")
    combined["status"] = np.where(
        combined["pc1_variance_explained"].ge(0.3)
        & combined["subsample_recovery_rate"].ge(0.7)
        & combined["early_repeat_preservation"].ge(0.5),
        "primary",
        "exploratory",
    )
    combined["status_reasons"] = [
        ";".join(
            reason
            for condition, reason in (
                (row.pc1_variance_explained < 0.3, "pc1_variance_explained_lt_0.3"),
                (row.subsample_recovery_rate < 0.7, "subsample_recovery_rate_lt_0.7"),
                (row.early_repeat_preservation < 0.5, "early_repeat_preservation_lt_0.5"),
            )
            if condition
        )
        for row in combined.itertuples(index=False)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    stability.to_csv(targets["stability"], sep="\t", index=False)
    module_hallmark_purity.to_csv(
        targets["hallmark_purity"],
        sep="\t",
        index=True,
    )
    combined.to_csv(targets["combined"], sep="\t", index=False)
    gene_sets.manifest.to_csv(targets["sources"], sep="\t", index=False)
    summary_lines = [
        "Gene module reduction audit",
        f"Modules: {len(combined)}",
        f"Primary modules: {int(combined['status'].eq('primary').sum())}",
        f"Exploratory modules: {int(combined['status'].eq('exploratory').sum())}",
        f"Hallmark terms: {module_hallmark_purity.shape[1]}",
        "Gene-set sources: " + ", ".join(gene_sets.manifest["source"].astype(str)),
        "All non-grey modules remain in the projection bundle; status is audit-only.",
    ]
    targets["summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return combined
