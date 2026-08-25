"""Compute healthy-control log2FC matrices from organ-wise RMA outputs."""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import IO, Sequence

import pandas as pd

from .common import BASELINE_COLUMNS, PROJECT_ROOT, parse_bool_series, require_columns


LOG2FC_MANIFEST_COLUMNS: set[str] = {
    "sample_id",
    "organ",
    "time_label",
    "administration_route",
    "is_control",
    "is_healthy_control",
    "include_in_rma",
    "baseline_group_id",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the log2FC command-line parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Compute log2FC from organ-wise RMA matrices."
    )
    parser.add_argument("--organ", choices=("liver", "kidney", "all"), default="all")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rma-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-controls", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=250)
    return parser


def add_log2fc_eligibility(
    manifest: pd.DataFrame, min_controls: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach baseline membership and log2FC eligibility to RMA samples.

    Args:
        manifest: Complete Sample ID manifest.
        min_controls: Minimum healthy controls per organ/time/route pool.

    Returns:
        ``(rma_sample_audit, baseline_group_audit)``.

    Raises:
        ValueError: If required columns are absent or ``min_controls`` is invalid.
    """

    require_columns(manifest, LOG2FC_MANIFEST_COLUMNS, "sample manifest")
    if min_controls < 1:
        raise ValueError("min_controls must be at least 1")
    audited = manifest.copy()
    for column in ("is_control", "is_healthy_control", "include_in_rma"):
        audited[column] = parse_bool_series(audited[column], column)
    audited = audited[audited["include_in_rma"]].copy()
    healthy = audited[audited["is_healthy_control"]].copy()
    grouped_controls = healthy.groupby(BASELINE_COLUMNS, sort=True)["sample_id"].agg(list)

    group_rows: list[dict[str, object]] = []
    for key, control_ids in grouped_controls.items():
        organ, time_label, route = key
        baseline_group_id = audited.loc[
            (audited["organ"] == organ)
            & (audited["time_label"] == time_label)
            & (audited["administration_route"] == route),
            "baseline_group_id",
        ].iloc[0]
        group_rows.append(
            {
                "baseline_group_id": baseline_group_id,
                "organ": organ,
                "time_label": time_label,
                "administration_route": route,
                "n_controls": len(control_ids),
                "meets_min_controls": len(control_ids) >= min_controls,
                "control_sample_ids": ";".join(control_ids),
            }
        )
    group_audit = pd.DataFrame(group_rows)
    count_map = {
        (row.organ, row.time_label, row.administration_route): int(row.n_controls)
        for row in group_audit.itertuples()
    }
    audited["baseline_n_controls"] = [
        count_map.get((row.organ, row.time_label, row.administration_route), 0)
        for row in audited.itertuples()
    ]
    audited["include_in_log2fc"] = audited["baseline_n_controls"].ge(min_controls)
    audited["log2fc_exclusion_reason"] = ""
    audited.loc[
        ~audited["include_in_log2fc"], "log2fc_exclusion_reason"
    ] = "baseline_pool_lt_min_controls"

    if not group_audit.empty:
        sample_counts = audited.groupby(BASELINE_COLUMNS, sort=True).agg(
            n_rma_samples=("sample_id", "size"),
            n_treated=("is_control", lambda values: int((~values).sum())),
        )
        group_audit = group_audit.merge(
            sample_counts.reset_index(), on=BASELINE_COLUMNS, how="left", validate="one_to_one"
        )
    return audited, group_audit


def compute_log2fc_chunk(
    expression: pd.DataFrame,
    sample_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute log2FC and baseline means for one gene chunk.

    Args:
        expression: Chunk indexed by gene ID with RMA Sample IDs as columns.
        sample_audit: RMA sample audit for one organ.

    Returns:
        ``(log2fc_chunk, baseline_reference_chunk)``.
    """

    eligible = sample_audit[sample_audit["include_in_log2fc"]].copy()
    output = pd.DataFrame(index=expression.index)
    baseline_reference = pd.DataFrame(index=expression.index)
    for baseline_group_id, group in eligible.groupby("baseline_group_id", sort=True):
        control_ids = group.loc[group["is_healthy_control"], "sample_id"].tolist()
        sample_ids = group["sample_id"].tolist()
        if not control_ids:
            raise ValueError(f"No healthy controls in eligible pool {baseline_group_id}")
        missing = sorted(set(control_ids + sample_ids).difference(expression.columns))
        if missing:
            raise ValueError(
                f"RMA expression is missing samples for {baseline_group_id}: {missing[:10]}"
            )
        baseline = expression[control_ids].mean(axis=1)
        baseline_reference[baseline_group_id] = baseline
        output[sample_ids] = expression[sample_ids].subtract(baseline, axis=0)
    ordered_ids = eligible["sample_id"].tolist()
    return output[ordered_ids], baseline_reference


def _read_manifest(path: Path) -> pd.DataFrame:
    """Read the Sample ID manifest.

    Args:
        path: Manifest path.

    Returns:
        Manifest DataFrame.
    """

    manifest = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    require_columns(manifest, LOG2FC_MANIFEST_COLUMNS, str(path))
    return manifest


def _write_chunk(
    handle: IO[str], frame: pd.DataFrame, include_header: bool
) -> None:
    """Write one indexed matrix chunk to an open TSV stream.

    Args:
        handle: Open text stream.
        frame: Matrix chunk indexed by gene ID.
        include_header: Whether to include the header row.
    """

    output = frame.copy()
    output.insert(0, "gene_id", output.index)
    output.to_csv(
        handle,
        sep="\t",
        index=False,
        header=include_header,
        float_format="%.8g",
        lineterminator="\n",
    )


def process_organ(
    organ: str,
    rma_path: Path,
    output_dir: Path,
    sample_audit: pd.DataFrame,
    chunk_size: int,
) -> None:
    """Compute and write log2FC outputs for one organ.

    Args:
        organ: Organ name.
        rma_path: Organ RMA matrix path.
        output_dir: log2FC output directory.
        sample_audit: Full RMA sample audit.
        chunk_size: Genes per streaming chunk.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    organ_audit = sample_audit[
        sample_audit["organ"].str.casefold().eq(organ.casefold())
    ].copy()
    if organ_audit.empty:
        raise ValueError(f"No RMA samples found for organ {organ}")

    header = pd.read_csv(rma_path, sep="\t", nrows=0).columns.tolist()
    if not header or header[0] != "gene_id":
        raise ValueError(f"RMA matrix must begin with gene_id: {rma_path}")
    matrix_ids = header[1:]
    expected_ids = organ_audit["sample_id"].tolist()
    if set(matrix_ids) != set(expected_ids) or len(matrix_ids) != len(expected_ids):
        missing = sorted(set(expected_ids).difference(matrix_ids))[:10]
        unexpected = sorted(set(matrix_ids).difference(expected_ids))[:10]
        raise ValueError(
            f"RMA/manifest Sample ID mismatch for {organ}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    organ_audit = organ_audit.set_index("sample_id").loc[matrix_ids].reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    log2fc_path = output_dir / f"{organ.lower()}_log2fc.tsv.gz"
    baseline_path = output_dir / f"{organ.lower()}_baseline_reference.tsv.gz"
    temporary_log2fc = log2fc_path.with_name(log2fc_path.name + ".tmp")
    temporary_baseline = baseline_path.with_name(baseline_path.name + ".tmp")
    seen_genes: set[str] = set()
    wrote_chunk = False
    try:
        with gzip.open(temporary_log2fc, "wt", newline="") as log2fc_handle, gzip.open(
            temporary_baseline, "wt", newline=""
        ) as baseline_handle:
            for chunk in pd.read_csv(rma_path, sep="\t", chunksize=chunk_size):
                gene_ids = chunk.pop("gene_id").astype(str)
                duplicate_genes = seen_genes.intersection(gene_ids)
                if duplicate_genes:
                    raise ValueError(
                        f"Duplicate gene IDs in {rma_path}: {sorted(duplicate_genes)[:10]}"
                    )
                seen_genes.update(gene_ids)
                expression = chunk.apply(pd.to_numeric, errors="raise")
                expression.index = gene_ids
                log2fc_chunk, baseline_chunk = compute_log2fc_chunk(
                    expression, organ_audit
                )
                _write_chunk(log2fc_handle, log2fc_chunk, include_header=not wrote_chunk)
                _write_chunk(baseline_handle, baseline_chunk, include_header=not wrote_chunk)
                wrote_chunk = True
        if not wrote_chunk:
            raise ValueError(f"RMA matrix contains no genes: {rma_path}")
        os.replace(temporary_log2fc, log2fc_path)
        os.replace(temporary_baseline, baseline_path)
    finally:
        temporary_log2fc.unlink(missing_ok=True)
        temporary_baseline.unlink(missing_ok=True)
    print(
        f"[{organ.upper()}] log2FC: {len(seen_genes):,} genes x "
        f"{int(organ_audit['include_in_log2fc'].sum()):,} samples"
    )
    print(f"[{organ.upper()}] Output: {log2fc_path}")


def run(args: argparse.Namespace) -> None:
    """Run log2FC computation from parsed arguments.

    Args:
        args: Parsed command-line arguments.
    """

    project_root = args.project_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else project_root / "data" / "expression" / "metadata" / "sample_id_manifest.tsv"
    )
    rma_dir = (
        args.rma_dir.expanduser().resolve()
        if args.rma_dir is not None
        else project_root / "data" / "expression" / "rma"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "expression" / "log2fc"
    )
    manifest = _read_manifest(manifest_path)
    sample_audit, baseline_audit = add_log2fc_eligibility(
        manifest, min_controls=args.min_controls
    )

    organs = ("liver", "kidney") if args.organ == "all" else (args.organ,)
    for organ in organs:
        process_organ(
            organ=organ,
            rma_path=rma_dir / f"{organ}_rma_log2.tsv.gz",
            output_dir=output_dir,
            sample_audit=sample_audit,
            chunk_size=args.chunk_size,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_audit.to_csv(
        output_dir / "log2fc_sample_manifest.tsv", sep="\t", index=False
    )
    baseline_audit.to_csv(output_dir / "baseline_groups.tsv", sep="\t", index=False)
    print(f"Sample audit: {output_dir / 'log2fc_sample_manifest.tsv'}")
    print(f"Baseline audit: {output_dir / 'baseline_groups.tsv'}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the log2FC entry point.

    Args:
        argv: Optional explicit argument vector.
    """

    parser = build_parser()
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
