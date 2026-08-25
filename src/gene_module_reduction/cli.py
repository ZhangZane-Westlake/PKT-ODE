"""Command-line interface for Gene selection and fixed WGCNA reduction."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final, Optional, Sequence

import numpy as np
import pandas as pd

from .audit import (
    calculate_module_stability,
    calculate_module_hallmark_purity,
    load_gene_sets,
    write_audit,
)
from .eligibility import (
    FIT_MODEL_TIMES,
    build_treatment_eligibility,
    filter_eligibility_for_compound,
    read_split_manifest,
    write_eligibility,
)
from .projection import (
    fit_fixed_projection,
    load_module_bundle,
    transform_samples,
    write_projection_fit,
    write_transformed_modules,
)
from .selection import SelectionConfig, select_genes, sha256_file, write_selection


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
WGCNA_SCRIPT: Final[Path] = Path(__file__).with_name("wgcna_fit.R")


def _safe_component(value: str, name: str) -> str:
    """Validate one output-directory component.

    Args:
        value: Requested component.
        name: Argument name for errors.

    Returns:
        Validated component.
    """

    if not value or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"{name} must contain only letters, numbers, dot, underscore, or hyphen")
    return value


def _run_root(args: argparse.Namespace) -> Path:
    """Resolve the isolated run directory.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Run/split output root.
    """

    output_root = args.output_root.expanduser().resolve()
    base = (
        output_root
        / _safe_component(args.run_name, "run-name")
        / args.organ
    )
    if args.compound is not None:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.compound.strip()).strip("-.")
        if not normalized:
            raise ValueError("compound must contain at least one letter or number")
        base = base / "compounds" / normalized.casefold()
    return base / _safe_component(args.split_id, "split-id")


def _parameter_token(value: float) -> str:
    """Format one numeric parameter as a path-safe compact token.

    Args:
        value: Numeric configuration value.

    Returns:
        Stable token without a decimal point or sign punctuation.
    """

    return format(value, ".6g").replace("-", "m").replace(".", "p")


def _reduction_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Return the WGCNA parameters that define one reduction configuration.

    Args:
        args: Parsed fit, transform, or audit arguments.

    Returns:
        Ordered parameter mapping used for naming and manifests.
    """

    return {
        "scale_free_r2": args.scale_free_r2,
        "min_mean_connectivity": args.min_mean_connectivity,
        "deep_split": args.deep_split,
        "min_module_size": args.min_module_size,
        "merge_cut_height": args.merge_cut_height,
        "max_block_size": args.max_block_size,
        "max_p_outliers": args.max_p_outliers,
        "seed": args.wgcna_seed,
    }


def _reduction_config_id(args: argparse.Namespace) -> str:
    """Build the automatic path component for one WGCNA configuration.

    Args:
        args: Parsed fit, transform, or audit arguments.

    Returns:
        Parameter-readable configuration identifier.
    """

    parameters = _reduction_parameters(args)
    return (
        f"r2-{_parameter_token(float(parameters['scale_free_r2']))}"
        f"_conn-{_parameter_token(float(parameters['min_mean_connectivity']))}"
        f"_ds-{int(parameters['deep_split'])}"
        f"_mms-{int(parameters['min_module_size'])}"
        f"_merge-{_parameter_token(float(parameters['merge_cut_height']))}"
        f"_block-{int(parameters['max_block_size'])}"
        f"_outlier-{_parameter_token(float(parameters['max_p_outliers']))}"
        f"_seed-{int(parameters['seed'])}"
    )


def _reduction_root(args: argparse.Namespace) -> Path:
    """Resolve the parameter-isolated WGCNA/projection/audit directory.

    Args:
        args: Parsed fit, transform, or audit arguments.

    Returns:
        Reduction configuration output root.
    """

    return _run_root(args) / "reductions" / _reduction_config_id(args)


def _default_log2fc(project_root: Path, organ: str) -> Path:
    """Return the established organ log2FC matrix path.

    Args:
        project_root: Repository root.
        organ: Kidney or liver.

    Returns:
        Default matrix path.
    """

    return project_root / "data" / "expression" / "log2fc" / f"{organ}_log2fc.tsv.gz"


def _read_eligible_samples(path: Path) -> pd.DataFrame:
    """Read prepared eligible sample metadata.

    Args:
        path: Eligible-sample TSV path.

    Returns:
        Sample metadata.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Missing eligibility artifact: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def _write_run_manifest(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    result_counts: dict[str, int],
    split_manifest: Optional[Path],
    resolved_compound: Optional[tuple[str, str]],
) -> None:
    """Write the schema-versioned run manifest.

    Args:
        path: Output JSON path.
        args: Parsed prepare arguments.
        manifest_path: Source log2FC manifest.
        result_counts: Eligibility counts.
        split_manifest: Optional compound split path.
        resolved_compound: Resolved compound number and name for a compound run.
    """

    if path.exists():
        raise FileExistsError(f"Run manifest already exists: {path}")
    payload: dict[str, object] = {
        "schema_version": 2,
        "organ": args.organ,
        "run_name": args.run_name,
        "split_id": args.split_id,
        "selection_scope": "single_compound" if args.compound is not None else "organ",
        "requested_compound": args.compound,
        "resolved_compound": (
            {
                "compound_no": resolved_compound[0],
                "compound_name": resolved_compound[1],
            }
            if resolved_compound is not None
            else None
        ),
        "eligibility_model_times": ["3H", "6H", "9H", "1D", "4D", "8D", "15D", "29D"],
        "fit_model_times": list(FIT_MODEL_TIMES),
        "expected_replicates": args.expected_replicates,
        "counts": result_counts,
        "inputs": {
            "manifest": f"<input>/{manifest_path.name}",
            "manifest_sha256": sha256_file(manifest_path),
            "split_manifest": (
                f"<input>/{split_manifest.name}" if split_manifest else None
            ),
            "split_manifest_sha256": sha256_file(split_manifest) if split_manifest else None,
        },
        "leakage_boundary": "15D/29D availability is eligibility-only; values are excluded from selection and WGCNA fitting",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_prepare(args: argparse.Namespace) -> int:
    """Run eight-timepoint treatment eligibility.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing log2FC manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t", low_memory=False)
    split_path = args.split_manifest.expanduser().resolve() if args.split_manifest else None
    split = read_split_manifest(split_path)
    result = build_treatment_eligibility(
        manifest=manifest,
        organ=args.organ,
        expected_replicates=args.expected_replicates,
        split_manifest=split,
    )
    resolved_compound: Optional[tuple[str, str]] = None
    if args.compound is not None:
        result, compound_no, compound_name = filter_eligibility_for_compound(
            result, args.compound
        )
        resolved_compound = (compound_no, compound_name)
        print(f"Resolved compound: {compound_name} ({compound_no})")
    n_eligible = int(result.treatment_summary["treatment_eligible"].sum())
    n_train = int(
        result.treatment_summary["treatment_eligible"]
        .astype(bool)
        .where(result.treatment_summary["partition"].eq("train"), False)
        .sum()
    )
    if args.compound is not None and n_train == 0:
        raise ValueError(
            "Single-compound selection requires at least one eligible treatment in "
            "the train partition"
        )
    n_fit_conditions = n_train * len(FIT_MODEL_TIMES)
    n_fit_samples = n_fit_conditions * args.expected_replicates
    print(f"Eligible {args.organ} treatments: {n_eligible}")
    print(f"Training treatments: {n_train}")
    print(f"Reduction-fit conditions: {n_fit_conditions}")
    print(f"Reduction-fit replicate samples: {n_fit_samples}")
    if args.dry_run:
        return 0
    run_root = _run_root(args)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_eligibility(result, run_root / "eligibility")
    _write_run_manifest(
        run_root / "run_manifest.json",
        args,
        manifest_path,
        {
            "eligible_treatments": n_eligible,
            "training_treatments": n_train,
            "reduction_fit_conditions": n_fit_conditions,
            "reduction_fit_samples": n_fit_samples,
            "eligible_samples": len(result.eligible_samples),
        },
        split_path,
        resolved_compound,
    )
    print(f"Output: {run_root}")
    return 0


def run_select(args: argparse.Namespace) -> int:
    """Run broad/specific Gene selection.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    run_root = _run_root(args)
    eligible_samples = _read_eligible_samples(
        run_root / "eligibility" / "eligible_samples.tsv.gz"
    )
    log2fc_path = (
        args.log2fc.expanduser().resolve()
        if args.log2fc is not None
        else _default_log2fc(args.project_root.expanduser().resolve(), args.organ)
    )
    config = SelectionConfig(
        broad_effect=args.broad_effect,
        strong_effect=args.strong_effect,
        broad_rate=(
            args.broad_rate
            if args.broad_rate is not None
            else (1.0 / 3.0 if args.compound is not None else 0.10)
        ),
        specific_rate=args.specific_rate,
        noise_floor=args.noise_floor,
        chunk_size=args.chunk_size,
    )
    result = select_genes(
        log2fc_path=log2fc_path,
        eligible_samples=eligible_samples,
        config=config,
        expected_replicates=args.expected_replicates,
        selection_mode="single_compound" if args.compound is not None else "organ",
    )
    write_selection(result, run_root / "selection")
    print(f"Input Genes: {len(result.gene_metrics)}")
    print(f"Selected Genes: {len(result.selected_gene_ids)}")
    print(f"Selection output: {run_root / 'selection'}")
    return 0


def _run_wgcna(
    args: argparse.Namespace,
    run_root: Path,
    reduction_root: Path,
) -> None:
    """Invoke the isolated R WGCNA runner.

    Args:
        args: Parsed fit-wgcna arguments.
        run_root: Shared eligibility/selection output root.
        reduction_root: Parameter-isolated reduction output root.

    Raises:
        RuntimeError: If Rscript is unavailable.
    """

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required for fit-wgcna")
    wgcna_directory = reduction_root / "wgcna"
    if wgcna_directory.exists() and any(wgcna_directory.iterdir()):
        raise FileExistsError(f"WGCNA directory is not empty: {wgcna_directory}")
    command = [
        rscript,
        str(WGCNA_SCRIPT),
        "--matrix",
        str(run_root / "selection" / "fit_replicate_values.tsv.gz"),
        "--output-dir",
        str(wgcna_directory),
        "--seed",
        str(args.wgcna_seed),
        "--threads",
        str(args.threads),
        "--scale-free-r2",
        str(args.scale_free_r2),
        "--min-mean-connectivity",
        str(args.min_mean_connectivity),
        "--deep-split",
        str(args.deep_split),
        "--min-module-size",
        str(args.min_module_size),
        "--merge-cut-height",
        str(args.merge_cut_height),
        "--max-block-size",
        str(args.max_block_size),
        "--max-p-outliers",
        str(args.max_p_outliers),
    ]
    subprocess.run(command, check=True)


def _validate_eigengene_alignment(
    wgcna_path: Path,
    sample_ids: list[str],
    module_ids: list[str],
    fixed_scores: np.ndarray,
) -> None:
    """Validate fixed PC1 scores against WGCNA eigengenes up to scale/sign.

    Args:
        wgcna_path: WGCNA eigengene TSV.
        sample_ids: Ordered reduction-fit replicate sample IDs.
        module_ids: Ordered fixed modules.
        fixed_scores: Sample-by-module fixed PC1 scores.
    """

    table = pd.read_csv(wgcna_path, sep="\t")
    if table["sample_id"].astype(str).tolist() != sample_ids:
        raise ValueError("WGCNA and fixed projection sample axes differ")
    for module_index, module_id in enumerate(module_ids):
        if module_id not in table.columns:
            raise ValueError(f"WGCNA eigengene output is missing {module_id}")
        correlation = float(
            np.corrcoef(table[module_id].to_numpy(dtype=float), fixed_scores[:, module_index])[0, 1]
        )
        if not np.isfinite(correlation) or abs(correlation) < 0.999:
            raise ValueError(
                f"Fixed projection does not reproduce WGCNA {module_id}: r={correlation}"
            )


def run_fit_wgcna(args: argparse.Namespace) -> int:
    """Fit WGCNA membership and the neutral fixed projection bundle.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    run_root = _run_root(args)
    reduction_root = _reduction_root(args)
    _run_wgcna(args, run_root, reduction_root)
    gene_modules_path = reduction_root / "wgcna" / "gene_modules.tsv"
    gene_modules = pd.read_csv(gene_modules_path, sep="\t", dtype={"gene_id": str})
    gene_metrics = pd.read_csv(
        run_root / "selection" / "gene_selection.tsv.gz", sep="\t", low_memory=False
    )
    fit_matrix_path = run_root / "selection" / "fit_replicate_values.npz"
    result = fit_fixed_projection(fit_matrix_path, gene_modules, gene_metrics)
    result.manifest["reduction_config_id"] = _reduction_config_id(args)
    result.manifest["reduction_parameters"] = _reduction_parameters(args)
    with np.load(fit_matrix_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str).tolist()
    _validate_eigengene_alignment(
        reduction_root / "wgcna" / "wgcna_module_eigengenes.tsv",
        sample_ids,
        result.bundle.module_ids,
        result.fit_sample_scores_raw,
    )
    write_projection_fit(result, sample_ids, reduction_root / "projection")
    print(f"Modules: {len(result.bundle.module_ids)}")
    print(f"Reduction config: {_reduction_config_id(args)}")
    print(f"Bundle: {reduction_root / 'projection' / 'module_bundle.npz'}")
    return 0


def run_transform(args: argparse.Namespace) -> int:
    """Apply a fixed projection to the complete eligible cohort.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    run_root = _run_root(args)
    reduction_root = _reduction_root(args)
    bundle = load_module_bundle(reduction_root / "projection" / "module_bundle.npz")
    sample_metadata = _read_eligible_samples(
        run_root / "eligibility" / "eligible_samples.tsv.gz"
    )
    log2fc_path = (
        args.log2fc.expanduser().resolve()
        if args.log2fc is not None
        else _default_log2fc(args.project_root.expanduser().resolve(), args.organ)
    )
    transformed = transform_samples(
        log2fc_path=log2fc_path,
        sample_metadata=sample_metadata,
        bundle=bundle,
        expected_replicates=args.expected_replicates,
        chunk_size=args.chunk_size,
    )
    write_transformed_modules(transformed, bundle, reduction_root / "projection")
    print(f"Projected samples: {len(transformed.sample_metadata)}")
    print(f"Projected treatments: {len(transformed.treatment_ids)}")
    return 0


def run_audit(args: argparse.Namespace) -> int:
    """Run module stability and Hallmark purity audits.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    run_root = _run_root(args)
    reduction_root = _reduction_root(args)
    gene_sets = load_gene_sets(
        args.gene_set_manifest.expanduser().resolve(),
    )
    bundle = load_module_bundle(reduction_root / "projection" / "module_bundle.npz")
    fit_sample_metadata = pd.read_csv(
        run_root / "selection" / "fit_sample_metadata.tsv", sep="\t", dtype=str
    )
    stability = calculate_module_stability(
        fit_matrix_path=run_root / "selection" / "fit_replicate_values.npz",
        fit_sample_metadata=fit_sample_metadata,
        bundle=bundle,
        n_subsamples=args.subsamples,
        compound_fraction=args.compound_fraction,
        recovery_cosine=args.recovery_cosine,
        seed=args.seed,
    )
    module_summary = pd.read_csv(
        reduction_root / "projection" / "module_projection_summary.tsv", sep="\t"
    )
    gene_modules = pd.read_csv(
        reduction_root / "projection" / "gene_modules.tsv.gz", sep="\t", dtype=str
    )
    module_hallmark_purity = calculate_module_hallmark_purity(
        gene_modules=gene_modules,
        gene_sets=gene_sets,
    )
    combined = write_audit(
        module_summary=module_summary,
        stability=stability,
        module_hallmark_purity=module_hallmark_purity,
        gene_sets=gene_sets,
        directory=reduction_root / "audit",
    )
    print(f"Primary modules: {int(combined['status'].eq('primary').sum())}")
    print(f"Exploratory modules: {int(combined['status'].eq('exploratory').sum())}")
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared run identity arguments.

    Args:
        parser: Parser to extend.
    """

    parser.add_argument("--organ", choices=("kidney", "liver"), required=True)
    parser.add_argument(
        "--compound",
        default=None,
        help=(
            "Exact compound name or compound_no. When set, use an isolated "
            "single-compound output tree and Broad-only selection."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-id", default="loose_stage")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "gene_module_reduction",
    )


def _add_reduction_identity_arguments(
    parser: argparse.ArgumentParser,
    allow_seed_alias: bool = False,
) -> None:
    """Add parameters that identify one WGCNA reduction configuration.

    Args:
        parser: Parser to extend.
        allow_seed_alias: Also accept the established ``--seed`` spelling.
    """

    seed_options = (
        ("--seed", "--wgcna-seed")
        if allow_seed_alias
        else ("--wgcna-seed",)
    )
    parser.add_argument(*seed_options, dest="wgcna_seed", type=int, default=42)
    parser.add_argument("--scale-free-r2", type=float, default=0.8)
    parser.add_argument("--min-mean-connectivity", type=float, default=5.0)
    parser.add_argument("--deep-split", type=int, default=2)
    parser.add_argument("--min-module-size", type=int, default=30)
    parser.add_argument("--merge-cut-height", type=float, default=0.25)
    parser.add_argument("--max-block-size", type=int, default=5000)
    parser.add_argument("--max-p-outliers", type=float, default=0.1)


def build_parser() -> argparse.ArgumentParser:
    """Create the Gene-module reduction CLI parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Training-only Gene selection and fixed WGCNA module reduction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build eight-timepoint eligibility")
    _add_run_arguments(prepare)
    prepare.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "expression" / "log2fc" / "log2fc_sample_manifest.tsv",
    )
    prepare.add_argument("--split-manifest", type=Path, default=None)
    prepare.add_argument("--expected-replicates", type=int, default=3)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(handler=run_prepare)

    select = subparsers.add_parser(
        "select", help="Select organ-level Broad/Specific or compound-level Broad Genes"
    )
    _add_run_arguments(select)
    select.add_argument("--log2fc", type=Path, default=None)
    select.add_argument("--expected-replicates", type=int, default=3)
    select.add_argument("--broad-effect", type=float, default=0.5)
    select.add_argument("--strong-effect", type=float, default=1.0)
    select.add_argument(
        "--broad-rate",
        type=float,
        default=None,
        help="Default: 0.10 for organ scope; 1/3 for single-compound scope",
    )
    select.add_argument("--specific-rate", type=float, default=1.0 / 3.0)
    select.add_argument("--noise-floor", type=float, default=0.1)
    select.add_argument("--chunk-size", type=int, default=250)
    select.set_defaults(handler=run_select)

    fit = subparsers.add_parser("fit-wgcna", help="Fit WGCNA and fixed PC1 loadings")
    _add_run_arguments(fit)
    _add_reduction_identity_arguments(fit, allow_seed_alias=True)
    fit.add_argument("--threads", type=int, default=1)
    fit.set_defaults(handler=run_fit_wgcna)

    transform = subparsers.add_parser("transform", help="Apply the fixed module bundle")
    _add_run_arguments(transform)
    _add_reduction_identity_arguments(transform)
    transform.add_argument("--log2fc", type=Path, default=None)
    transform.add_argument("--expected-replicates", type=int, default=3)
    transform.add_argument("--chunk-size", type=int, default=250)
    transform.set_defaults(handler=run_transform)

    audit = subparsers.add_parser(
        "audit",
        help="Audit stability and module Hallmark purity",
    )
    _add_run_arguments(audit)
    _add_reduction_identity_arguments(audit)
    audit.add_argument(
        "--gene-set-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "gene_sets"
            / "gene_set_manifest.hallmark.tsv"
        ),
    )
    audit.add_argument("--subsamples", type=int, default=20)
    audit.add_argument("--compound-fraction", type=float, default=0.8)
    audit.add_argument("--recovery-cosine", type=float, default=0.7)
    audit.add_argument("--seed", type=int, default=42)
    audit.set_defaults(handler=run_audit)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected Gene-module reduction stage.

    Args:
        argv: Optional argument tokens.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    return int(args.handler(args))
