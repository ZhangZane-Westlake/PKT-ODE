"""Build the Open TG-GATEs Sample ID manifest and RMA filtering audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .common import (
    PROJECT_ROOT,
    build_dataset_summary,
    build_sample_manifest,
    index_cel_files,
    read_attribute_table,
    read_pathology_table,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Build Sample IDs and rat-level RMA inclusion metadata."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Open TG-GATEs raw root; defaults to <project-root>/data/raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Metadata output directory; defaults to data/expression/metadata.",
    )
    parser.add_argument("--min-controls", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing output files.",
    )
    return parser


def summarize_for_console(manifest: pd.DataFrame) -> str:
    """Format the key manifest counts for terminal output.

    Args:
        manifest: Complete sample manifest.

    Returns:
        Multi-line count summary.
    """

    input_controls = int(manifest["is_control"].sum())
    input_treated = len(manifest) - input_controls
    rma = manifest[manifest["include_in_rma"]]
    log2fc = manifest[manifest["include_in_log2fc"]]
    lines = [
        f"Input samples: {len(manifest):,}",
        f"  Treated: {input_treated:,}",
        f"  Control: {input_controls:,}",
        f"SP-rat mapped samples: {int(manifest['has_sp_pathology'].sum()):,}",
        f"RMA samples: {len(rma):,}",
        f"  Treated: {int((~rma['is_control']).sum()):,}",
        f"  Healthy control: {int(rma['is_healthy_control'].sum()):,}",
        f"Planned log2FC samples: {len(log2fc):,}",
        f"  Treated: {int((~log2fc['is_control']).sum()):,}",
        f"  Healthy control: {int(log2fc['is_healthy_control'].sum()):,}",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and optionally write manifest artifacts.

    Args:
        args: Parsed command-line arguments.

    Returns:
        ``(manifest, dataset_summary)``.
    """

    project_root = args.project_root.expanduser().resolve()
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else project_root / "data" / "raw"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "expression" / "metadata"
    )
    attributes = read_attribute_table(raw_dir / "Open-tggates_AllAttribute.tsv")
    pathology = read_pathology_table(raw_dir / "open_tggates_pathology.csv")
    cel_paths = index_cel_files(raw_dir)
    manifest = build_sample_manifest(
        attributes=attributes,
        pathology=pathology,
        cel_paths=cel_paths,
        project_root=project_root,
        min_controls=args.min_controls,
    )
    summary = build_dataset_summary(manifest)
    print(summarize_for_console(manifest))
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_dir / "sample_id_manifest.tsv", sep="\t", index=False)
        summary.to_csv(output_dir / "dataset_summary.tsv", sep="\t", index=False)
        print(f"Manifest: {output_dir / 'sample_id_manifest.tsv'}")
        print(f"Summary: {output_dir / 'dataset_summary.tsv'}")
    return manifest, summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Sample ID and RMA filtering entry point.

    Args:
        argv: Optional explicit argument vector.
    """

    parser = build_parser()
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
